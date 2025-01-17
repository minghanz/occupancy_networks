import yaml
from torchvision import transforms
from im2mesh import data
from im2mesh import onet, r2n2, psgn, pix2mesh, dmc
from im2mesh import preprocess
import torch

method_dict = {
    'onet': onet,
    'r2n2': r2n2,
    'psgn': psgn,
    'pix2mesh': pix2mesh,
    'dmc': dmc,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False, data_key='data'):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg[data_key]['dataset']
    dataset_folder = cfg[data_key]['path']

    if mode == 'test':
        reg_mode = cfg['test'].get('reg', True)
    else:
        reg_mode = cfg['training'].get('dual', False)
    reg_benchmark_mode = mode == 'test' and cfg['test'].get('reg_benchmark', False)
    duo_mode = cfg[data_key]['duo_mode']

    bench_input_folder = cfg[data_key].get('path_bench', None) if reg_benchmark_mode else None
    
    # Create dataset
    if dataset_type == 'modelnet':
        dataset = data.fmr_get_dataset(cfg[data_key], mode)
    elif dataset_type == 'Shapes3D':
        categories = cfg[data_key]['classes']

        # Get split
        splits = {
            'train': cfg[data_key]['train_split'],
            'val': cfg[data_key]['val_split'],
            'test': cfg[data_key]['test_split'],
        }
        split = splits[mode]

        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if reg_benchmark_mode:
            assert isinstance(inputs_field, list)
            fields['inputs'] = inputs_field[0]
            fields['inputs_2'] = inputs_field[1]
            fields['T21'] = inputs_field[2]
        elif duo_mode:
            assert isinstance(inputs_field, list)
            fields['inputs'] = inputs_field[0]
            fields['T'] = inputs_field[1]
        elif inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            bench_input_folder=bench_input_folder,
        )
        if reg_mode:
            rot_magmax = cfg[data_key]['rotate_test'] if mode == 'test' else cfg[data_key]['rotate']
            resamp_mode = cfg[data_key]['resamp_mode']
            pcl_noise = cfg[data_key]['pointcloud_noise']
            dataset = data.PairedDataset(dataset, rot_magmax, duo_mode, reg_benchmark_mode, resamp_mode, pcl_noise)

    elif dataset_type == 'kitti':
        dataset = data.KittiDataset(
            dataset_folder, img_size=cfg[data_key]['img_size'],
            return_idx=return_idx
        )
    elif dataset_type == 'online_products':
        dataset = data.OnlineProductDataset(
            dataset_folder, img_size=cfg[data_key]['img_size'],
            classes=cfg[data_key]['classes'],
            max_number_imgs=cfg['generation']['max_number_imgs'],
            return_idx=return_idx, return_category=return_category
        )
    elif dataset_type == 'images':
        dataset = data.ImageDataset(
            dataset_folder, img_size=cfg[data_key]['img_size'],
            return_idx=return_idx,
        )
    elif dataset_type == '7scene':
        if mode in ['train', 'val']:
            if mode == 'train':
                dataset, _ = data.get_datasets_7scenes(cfg[data_key], mode)
            else:
                _, dataset = data.get_datasets_7scenes(cfg[data_key], mode)
        else:
            dataset = data.get_datasets_7scenes(cfg[data_key], mode)
    else:
        raise ValueError('Invalid dataset "%s"' % cfg[data_key]['dataset'])
 
    if data_key == 'data' and 'data2' in cfg and mode == 'train':
        dataset2 = get_dataset(mode, cfg, return_idx, return_category, data_key='data2')
        dataset = [dataset, dataset2]

    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']
    with_transforms = cfg['data']['with_transforms']
    if mode == 'test':
        reg_mode = cfg['test'].get('reg', True)
    else:
        reg_mode = cfg['training'].get('dual', False)
    reg_benchmark_mode = mode == 'test' and cfg['test'].get('reg_benchmark', False)
    duo_mode = cfg['data']['duo_mode']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = data.ImagesField(
            cfg['data']['img_folder'], transform,
            with_camera=with_camera, random_view=random_view
        )
    elif input_type == 'pointcloud':
        if reg_benchmark_mode:
            inputs_field_1 = data.PointCloudField(
                cfg['data']['pointcloud_file_bench_1'], transform=None,
                with_transforms=False
            )
            inputs_field_2 = data.PointCloudField(
                cfg['data']['pointcloud_file_bench_2'], transform=None,
                with_transforms=False
            )
            T_field_21 = data.RotationField(
                cfg['data']['T21_file']
            )
            inputs_field = [inputs_field_1, inputs_field_2, T_field_21]
        else:
            if mode == 'train': 
                pointcloud_n = cfg['data']['pointcloud_n']
            else:
                pointcloud_n = cfg['data']['pointcloud_n_val']
            assert reg_mode
            if reg_mode:
                transform = data.SubsamplePointcloud(pointcloud_n)
            else:
                transform = transforms.Compose([
                    data.SubsamplePointcloud(pointcloud_n),
                    data.PointcloudNoise(cfg['data']['pointcloud_noise'])
                ])
            with_transforms = cfg['data']['with_transforms']
            inputs_field = data.PointCloudField(
                cfg['data']['pointcloud_file'], transform,
                with_transforms=with_transforms
            )
            if duo_mode:
                T_field = data.TransformationField(cfg['data']['T_file'])
                inputs_field = [inputs_field, T_field]

    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field


def get_preprocessor(cfg, dataset=None, device=None):
    ''' Returns preprocessor instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset
        device (device): pytorch device
    '''
    p_type = cfg['preprocessor']['type']
    cfg_path = cfg['preprocessor']['config']
    model_file = cfg['preprocessor']['model_file']

    if p_type == 'psgn':
        preprocessor = preprocess.PSGNPreprocessor(
            cfg_path=cfg_path,
            pointcloud_n=cfg['data']['pointcloud_n'],
            dataset=dataset,
            device=device,
            model_file=model_file,
        )
    elif p_type is None:
        preprocessor = None
    else:
        raise ValueError('Invalid Preprocessor %s' % p_type)

    return preprocessor
