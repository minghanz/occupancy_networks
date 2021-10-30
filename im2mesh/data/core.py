import os
import logging
from torch.utils.data import Dataset, dataloader
import numpy as np
import yaml

from im2mesh.data.transforms import gen_randrot, apply_transformation, apply_rot, totensor_inplace, NoisePairBatchIP
from fmr_transforms import OnUnitCube
import torch

logger = logging.getLogger(__name__)


# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError

class PairedDataset(Dataset):
    '''Given a dataset that spit one sample at a time, return a pair, 
    so that they are related by a rigid body transformation corrupted by some noise
    (e.g. Gaussian noise, resampling, density difference). 

    For training: 
    Max number resampling is per instance. 

    Gaussian noise is per pair. (applied per instance but it takes place at PairedDataset because we may want different noise on the same sampled point cloud)
    Generation of rigid body transformation is per pair. 

    Resampling for randomness in number of points is per batch. (so that we can have randomness in number of points but consistent in a batch)
    Application of rigid body transformation is per pair after generating the generation or per batch. (per batch so that the operation of transformation is on the same device as registration)
    Optional centering is per batch. (after all sampling)

    For testing:
    No resampling or Gaussian noise. 
    Load Rigid body transformation from file, per pair. '''
    def __init__(self, dataset, rot_magmax=None, duo_mode=False, reg_benchmark_mode=False, resamp_mode=True, pcl_noise=None) -> None:
        '''
        Args:
            dataset: the Dataset object that give one instance as a time
            duo_mode: if True, the output pair is formed using two instances from different indices
            (reg_mode: True when we use this PairedDataset.
             default_mode: inputs and inputs_2 from the same file
             reg_benchmark_mode: inputs and inputs_2 from different files of the same indices, 
              need a transformation for a pair. [done outside of this dataset]
             duo_mode: inputs and inputs_2 from the same file of different indices, 
              need a transformation for each index. 
             duo_benchmark_mode?
             )
            transform: if not None, each pair goes through the transforms specified
        '''
        super().__init__()
        self.dataset = dataset
        self.rot_magmax = rot_magmax
        self.duo_mode = duo_mode
        self.reg_benchmark_mode = reg_benchmark_mode
        self.resamp_mode = resamp_mode
        self.pcl_noise = pcl_noise
        self.noise_op = NoisePairBatchIP(pcl_noise) if pcl_noise > 0 else None
        self.unitcube_op = OnUnitCube()

    def __len__(self):
        return len(self.dataset)
    def get_model_dict(self, idx):
        return self.dataset.get_model_dict(idx)
        
    def duo_load(self, idx, data):
        '''Load the data of adjacent index. '''
        if idx < len(self)-1:
            idx_2 = idx + 1
        else:
            idx_2 = idx - 1
        data_2 = self.dataset[idx_2]
        totensor_inplace(data_2)
        for key, value in data_2:
            dotidx = key.find('.')
            key_2 = key + '_2' if dotidx == -1 else key[:dotidx] + '_2' + key[dotidx:]
            data[key_2] = value
        return

    def duo_preprocess(self, data):
        '''Put inputs and inputs_2 in the same reference frame (the frame of inputs_2), 
        and then put both into a unit cube together (using the same spec).
        '''
        ### Put inputs and inputs_2 in the same reference frame
        T01 = data['T']
        T02 = data['T_2']
        T21 = torch.matmul(torch.inverse(T02), T01)
        data['inputs_rawT'] = data['inputs'].clone()
        data['inputs'] = apply_transformation(T21, data['inputs'])  
        if 'points' in data:
            data['points'] = apply_transformation(T21, data['points'])
        ### Centralize inputs and inputs_2 to unit cube using the same spec
        data['inputs'], spec = self.unitcube_op(data['inputs'], return_spec=True)
        data['inputs_2'] = self.unitcube_op(data['inputs_2'], spec)
        if 'points' in data:
            data['points'] = self.unitcube_op(data['points'], spec)
            data['points_2'] = self.unitcube_op(data['points_2'], spec)
        return

    def __getitem__(self, idx):
        data = self.dataset[idx]
        totensor_inplace(data)
        if self.reg_benchmark_mode:
            assert 'inputs_2' in data and 'T21' in data, "{}".format(list(data.keys()))
        else:
            if self.duo_mode:
                self.duo_load(idx, data)
                self.duo_preprocess(data)
                assert 'inputs_2' in data and 'T21' not in data
            else:
                ### when not duo_mode or reg_benchmark_mode (which both gives 'inputs_2' item)
                if self.resamp_mode:
                    self.dataset.load_field(idx, data, 'inputs_2', self.dataset.fields['inputs'])
                else:
                    data['inputs_2'] = data['inputs'].clone()
                data['points_2'] = data['points'].clone()
                totensor_inplace(data)
                assert 'inputs_2' in data and 'T21' not in data

            if 'T21' not in data:
                rotmat, deg = gen_randrot(self.rot_magmax)
                data['T21'] = rotmat
                data['T21.deg'] = deg

            if self.noise_op is not None:
                self.noise_op(data)

        # ### rotate one of the pair
        # data['inputs_2'] = apply_rot(data['T21'], data['inputs'])
    
        # data['inputs_3'] = apply_rot(data['T21'], data['inputs'])
        # diff_pts_rmse = torch.norm(data['inputs_3'] - data['inputs_2'], dim=1).mean()
        # diff_pts_rmse1 = torch.norm(data['inputs_2'], dim=1).mean()
        # diff_pts_rmse2 = torch.norm(data['inputs_3'], dim=1).mean()
        # logging.info("diff_pts_rmse dataset %.4f %.4f %.4f"%(diff_pts_rmse.item(), diff_pts_rmse1.item(), diff_pts_rmse2.item() ) )
        # logging.info("data['inputs'].dtype {} shape {}, device {}".format(data['inputs'].dtype, data['inputs'].shape, data['inputs'].device))
        # logging.info("data['inputs_2'].dtype {} shape {}, device {}".format(data['inputs_2'].dtype, data['inputs_2'].shape, data['inputs_2'].device))
        
        # if 'points' in data:
        #     data['points_2'] = apply_rot(data['T21'], data['points'])
        return data
        
        

class Shapes3dDataset(Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None, bench_input_folder=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform

        self.bench_input_folder = bench_input_folder if bench_input_folder is not None else dataset_folder    # the bench input root may be different

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        ### for ModelNet40, there is no metadata_file
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m}
                for m in models_c
            ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def load_field(self, idx, data, field_name, field, category=None, model=None, c_idx=None):
        if category is None:
            category = self.models[idx]['category']
            model = self.models[idx]['model']
            c_idx = self.metadata[category]['idx']
       
        dataset_folder = self.bench_input_folder if 'inputs' in field_name or 'T21' in field_name else self.dataset_folder

        model_path = os.path.join(dataset_folder, category, model)
        try:
            field_data = field.load(model_path, idx, c_idx)
        except Exception as e:
            if self.no_except:
                logger.warn(
                    'Error occured when loading field %s of model %s'
                    % (field_name, model)
                )
                logger.warn(e)
                return None
            else:
                raise

        if isinstance(field_data, dict):
            for k, v in field_data.items():
                if k is None:
                    data[field_name] = v
                else:
                    data['%s.%s' % (field_name, k)] = v
        else:
            data[field_name] = field_data

        return

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        data = {}

        for field_name, field in self.fields.items():
            self.load_field(idx, data, field_name, field, category, model, c_idx)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
