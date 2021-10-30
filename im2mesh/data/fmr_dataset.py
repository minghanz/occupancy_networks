from posixpath import splitext
import torch
from torch.utils.data import Dataset
import torchvision

import os
import glob
import copy
import six
import numpy as np

import fmr_mesh as mesh
import fmr_transforms as transforms
from register_util import angle_of_R
from im2mesh.data.transforms import gen_randrot, apply_transformation, apply_rot

def get_categories(categoryfile):
    cinfo = None
    if categoryfile:
        categories = [line.rstrip('\n') for line in open(categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    return cinfo

# find the total class names and its corresponding index from a folder
# (see the data storage structure of modelnet40)
def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# get the whole 3D point cloud paths for a given class
def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []
    model_dicts = []

    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
                model_dicts.append({'category': target, 'model': os.path.splitext(os.path.basename(path))[0]})
    return samples, model_dicts

def T44_from_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
        linesmat = lines[1:5]
        mat = [[float(x) for x in line.split()] for line in linesmat]
        mat = np.array(mat)
        # print(mat.shape, mat) #4*4
    return mat

class PointCloudDataset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """

    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        # find all the class names
        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        # get all the 3D point cloud paths for the class of class_to_idx
        samples, model_dicts = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.samples = samples

        self.dataset_folder = rootdir
        # print(self.samples)   # [(path_of_model, class_idx),]
        self.models = model_dicts
        # # Get all models
        # self.models = []
        # for c_idx, c in enumerate(classes):
        #     subpath = os.path.join(rootdir, c)
        #     if not os.path.isdir(subpath):
        #         print('Category %s does not exist in dataset.' % c)

        #     split_file = os.path.join(subpath, split + '.lst')
        #     with open(split_file, 'r') as f:
        #         models_c = f.read().split('\n')
            
        #     self.models += [
        #         {'category': c, 'model': m}
        #         for m in models_c
        #     ]

    def get_model_dict(self, idx):
        return self.models[idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        :param index:
        :return:
        """
        path, target = self.samples[index]
        try:
            sample = self.fileloader(path)
        except Exception as e:
            print(e)
            return self.__getitem__(index+1)

        txt = path.replace('.ply', '.info.txt')
        mat = T44_from_txt(txt)
        mat = torch.from_numpy(mat).to(dtype=torch.float)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, mat

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2


class ModelNet(PointCloudDataset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None, is_uniform_sampling=False):
        # if you would like to uniformly sampled points from mesh, use this function below
        if is_uniform_sampling:
            loader = mesh.offread_uniformed # used uniformly sampled points.
        else:
            loader = mesh.offread # use the original vertex in the mesh file
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None, resampling=False, duo_mode=False, noise=0):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

        self.rigid_transform_both = transforms.RandomTransformSE3(180, True, 0)

        self.resampling = resampling
        self.duo_mode = duo_mode
        self.noise = noise

    def __len__(self):
        return len(self.dataset)

    def get_model_dict(self, idx):
        return self.dataset.get_model_dict(idx)

    def __getitem__(self, index):
        pm, _, mat1 = self.dataset[index]

        ### possible resampling
        if self.duo_mode:
            if index < len(self) - 1:
                p0, _, mat0 = self.dataset[index+1]
            else:
                p0, _, mat0 = self.dataset[index-1]
        elif self.resampling:
            p0, _, _ = self.dataset[index]
            # print(p0-pm)
        else:
            if self.template_modifier is not None:
                p0 = self.template_modifier(pm)
            else:
                p0 = pm.clone()

        ### common_transform
        common_transform = self.rigid_transform_both.generate_transform()
        p0 = self.rigid_transform_both.apply_transform(p0, common_transform)
        pm = self.rigid_transform_both.apply_transform(pm, common_transform)

        ### duo mode alignment
        if self.duo_mode:
            mat01 = torch.matmul(torch.inverse(mat1), mat0)

            p0 = torch.matmul(mat01[:3, :3], p0.transpose(0,1)) + mat01[:3, [3]]
            p0 = p0.transpose(0,1)

        ### relative transformation
        rot_magmax = 180
        rotmat, deg = gen_randrot(rot_magmax)
        # p1 = apply_rot(rotmat, pm)
        p1 = pm.clone()

        # if self.source_modifier is not None:
        #     p_ = self.source_modifier(pm)
        #     p1 = self.rigid_transform(p_)
        # else:
        #     p1 = self.rigid_transform(pm)
        # igt = self.rigid_transform.igt

        ### gaussian noise
        if self.noise > 0:
            p1 = torch.tensor(np.float32(np.random.normal(p1, self.noise)))
            p0 = torch.tensor(np.float32(np.random.normal(p0, self.noise)))

        # p0: template, p1: source, igt: transform matrix from p0 to p1

        data = dict()
        data['inputs'] = p0
        data['inputs_2'] = p1
        data['T21'] = rotmat #igt[:3, :3]
        data['T21.deg'] = deg #angle_of_R(data['T21'])
        data['idx'] = index
        
        # return p0, p1, igt
        return data

def fmr_get_dataset(cfg_data, mode):
# global dataset function, could call to get dataset
    dataset_type = cfg_data['dataset']
    if dataset_type == 'modelnet':
        # if args.mode == 'train':
        if True:
            # set path and category file for training
            npt_key = 'pointcloud_n_val' if mode == 'test' else 'pointcloud_n'
            num_points = cfg_data[npt_key]
            # args.categoryfile = './data/categories/modelnet40_half1.txt'
            # args.categoryfile = './data/categories/modelnet40.txt'
            cinfo = get_categories(cfg_data.get('category_file', '') )
            assert cinfo is not None, cfg_data.get('category_file', '')
            transform = torchvision.transforms.Compose([ \
                transforms.Mesh2Points(), \
                transforms.OnUnitCube(), \
                transforms.Resampler(num_points), \
                ])

            # args.dataset_path = './data/ModelNet40'
            dataset_path = cfg_data['path']
            is_uniform_sampling = cfg_data.get('uniform_sampling', True)
            ### always use uniform sampling, because otherwise the point cloud is in abnormal shape
            if mode == 'train':
                dataset_one = ModelNet(dataset_path, train=1, transform=transform, classinfo=cinfo, is_uniform_sampling=is_uniform_sampling)
            else:
                dataset_one = ModelNet(dataset_path, train=0, transform=transform, classinfo=cinfo, is_uniform_sampling=is_uniform_sampling)

            mag_randomly = True #False #True
            rot_mag_key = 'rotate_test' if mode == 'test' else 'rotate'
            rot_mag = cfg_data[rot_mag_key]
            trans_mag = 0
            resampling = cfg_data['resamp_mode']
            noise = cfg_data['pointcloud_noise']

            dataset = TransformedDataset(dataset_one, transforms.RandomTransformSE3(rot_mag, mag_randomly, trans_mag), resampling=resampling, noise=noise)
            # return trainset, testset
            return dataset