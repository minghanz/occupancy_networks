from numpy.core.records import _OrderedCounter
import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid

### https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
import logging

from geometry import PointLK

import numpy as np
try:
    from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
except Exception as e:
    print(e)
    print("pytorch 3d may not be installed. You may not be able to run register.py on this machine. ")

from register_util import *

from im2mesh.data.transforms import apply_rot, SubSamplePairBatchIP, CentralizePairBatchIP, RotatePairBatchIP

def get_parser():

    parser = argparse.ArgumentParser(
        description='Extract meshes from occupancy process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    return parser

def setup_logging(generation_dir):
    level    = logging.INFO
    format   = '%(asctime)s %(message)s'
    datefmt = '%m-%d %H:%M:%S'
    handlers = [logging.FileHandler(os.path.join(generation_dir, 'msgs.log')), logging.StreamHandler()]

    logging.basicConfig(level = level, format = format, datefmt=datefmt, handlers = handlers, )
    logging.info('Hey, logging is written to {}!'.format(os.path.join(generation_dir, 'msgs.log')))
    return

def setup_f_out(cfg):
    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])

    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)

    setup_logging(generation_dir)
    return out_dir, generation_dir

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir, generation_dir = setup_f_out(cfg)
    
    # out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    # out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    batch_size = cfg['generation']['batch_size']
    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None:
        vis_n_outputs = -1

    # Dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=3, shuffle=False)
    
    centralize = cfg['generation']['centralize']
    n1 = cfg['generation']['n1']
    n2 = cfg['generation']['n2']
    sub_op = SubSamplePairBatchIP(n1, n2, n2, device)
    rotate_op = RotatePairBatchIP()
    ctr_op = CentralizePairBatchIP()
    noise = cfg['data']['pointcloud_noise']
    resamp = cfg['data']['resamp_mode']

    for it, data in enumerate(tqdm(test_loader)):

        ### process the directory
        # Get index etc.
        idx = data['idx'].item()

        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}
        
        modelname = model_dict['model']
        category_id = model_dict.get('category', 'n/a')

        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        if category_id != 'n/a':
            folder_name = str(category_id)
            if category_name != 'n/a':
                folder_name = str(folder_name) + '_' + category_name.split(',')[0]

            generation_dir_cat = os.path.join(generation_dir, folder_name)
            
            generation_dir_model = os.path.join(generation_dir_cat, modelname)

            if not os.path.exists(generation_dir_model):
                os.makedirs(generation_dir_model)

        # ### check whether the split files are copied to the target folder
        # split_f = dict()
        # ori_split_f = dict()
        # for key in ['train', 'val', 'test']:
        #     split_f[key] = os.path.join(generation_dir_cat, '{}.lst'.format(key))
        #     ori_split_f[key] = os.path.join(cfg['data']['path'], folder_name, '{}.lst'.format(key))
        #     if not os.path.exists(split_f[key]):
        #         shutil.copy(ori_split_f[key], split_f[key] )
        
        ### process the data
        sub_op(data)
        rotate_op(data)
        if centralize:
            ctr_op(data)

        input_1 = data['inputs'].cpu().numpy()[0]
        input_2_rot = data['inputs_2'].cpu().numpy()[0]
        R_gt = data['T21'].cpu().numpy()[0]
        rotdeg = data['T21.deg'][0]
        R21_dict = {'T': R_gt, 'deg': rotdeg}

        ctr_text = 'ctrd' if centralize else 'nctrd'
        resamp_text = 'resamp' if resamp else 'nresamp'

        T21_path = os.path.join(generation_dir_model, 'test_%d_%d_%s_%s_%s_R21.npz'%(n1, n2, str(noise).split('.')[1], ctr_text, resamp_text))
        pcl1_path = os.path.join(generation_dir_model, 'test_%d_%d_%s_%s_%s_pcl_1.npy'%(n1, n2, str(noise).split('.')[1], ctr_text, resamp_text))
        pcl2_path = os.path.join(generation_dir_model, 'test_%d_%d_%s_%s_%s_pcl_2.npy'%(n1, n2, str(noise).split('.')[1], ctr_text, resamp_text))

        with open(T21_path, 'wb') as f:
            np.savez(f, **R21_dict)

        with open(pcl1_path, 'wb') as f:
            np.save(f, input_1)

        with open(pcl2_path, 'wb') as f:
            np.save(f, input_2_rot)
            
