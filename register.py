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

import numpy as np
try:
    from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
except Exception as e:
    print(e)
    print("pytorch 3d may not be installed. You may not be able to run register.py on this machine. ")
def solve_R(f1, f2):
    """f1 and f2: (b*)m*3
    only work for batch_size=1
    """
    S = torch.matmul(f1.transpose(-1, -2), f2)  # 3*3
    U, sigma, V = torch.svd(S)
    R = torch.matmul(V, U.transpose(-1, -2))
    det = torch.det(R)
    # print(R)
    diag_1 = torch.tensor([1, 1, 0], device=R.device, dtype=R.dtype)
    diag_2 = torch.tensor([0, 0, 1], device=R.device, dtype=R.dtype)
    det_mat = torch.diag(diag_1 + diag_2 * det)

    # det_mat = torch.eye(3, device=R.device, dtype=R.dtype)
    # det_mat[2, 2] = det

    det_mat = det_mat.unsqueeze(0)
    # print(det_mat)
    R = torch.matmul(V, torch.matmul(det_mat, U.transpose(-1, -2)))
    print(det)
    # print(V.shape)
    
    return R
    

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=3, shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)

# accuracy
print("cfg['data']['pointcloud_n']", cfg['data']['pointcloud_n'])
print("angle_avg, trans_avg, angle180_avg, lower90, higher90")
metric_dict = dict()
metric_dict['angle_sum'] = 0
metric_dict['angle180_sum'] = 0
metric_dict['trans_sum'] = 0
metric_dict['num'] = 0
metric_dict['num_90-'] = 0
metric_dict['num_90+'] = 0
if not os.path.exists(generation_dir):
    os.makedirs(generation_dir)
f_res = open(os.path.join(generation_dir, "results.txt"), "w")

for it, data in enumerate(tqdm(test_loader)):

    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )

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
        mesh_dir = os.path.join(mesh_dir, str(category_id))
        pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    # Also copy ground truth
    if cfg['generation']['copy_groundtruth']:
        modelpath = os.path.join(
            dataset.dataset_folder, category_id, modelname, 
            cfg['data']['watertight_file'])
        out_file_dict['gt'] = modelpath

    register = True
    if register:
        out_1, out_2_rot, rot_d, pts_d = generator.generate_latent_conditioned(data)
        input_1, input_2_rot, input_2 = pts_d['inputs_1'], pts_d['inputs_rot_2'], pts_d['inputs_2']
        # print("out_1.shape", out_1.shape)    # 1, 513
        batch_size = out_1.shape[0]
        out_1 = out_1.reshape(batch_size, -1, 3)
        out_2_rot = out_2_rot.reshape(batch_size, -1, 3)

        out_1 = out_1.cpu().detach()
        out_2_rot = out_2_rot.cpu().detach()

        out_1_rot = rot_d['trot'].transform_points(out_1)
        diff_gt  = out_1_rot - out_2_rot
        diff_ori = out_1 - out_2_rot

        ### estimate rotation
        # Rs = solve_R(out_1, out_2_rot)
        Rs = solve_R(out_2_rot, out_1)
        trot = Rotate(R=Rs)
        out_1_rot_est = trot.transform_points(out_1)
        diff_est = out_1_rot_est - out_2_rot

        input_1_rot_est = trot.transform_points(input_1.cpu())

        ### estimate translation
        t_est = pts_d['t_2'] - trot.transform_points(pts_d['t_1'])
        input_1_trans_est = input_1_rot_est + t_est

        diff_gt = diff_gt.cpu().detach()
        diff_ori = diff_ori.cpu().detach()
        diff_est = diff_est.cpu().detach()

        diff_gt_inf = torch.abs(diff_gt).max().item()
        diff_ori_inf = torch.abs(diff_ori).max().item()
        diff_est_inf = torch.abs(diff_est).max().item()

        diff_gt_l2 = torch.norm(diff_gt).item()
        diff_ori_l2 = torch.norm(diff_ori).item()
        diff_est_l2 = torch.norm(diff_est).item()

        ### previous: gt, ori, est
        print("feat_diff_Linf (ori, est, gt) %.4f %.4f %.4f"%(diff_ori_inf, diff_est_inf, diff_gt_inf) )
        print("feat_diff_L2 (ori, est, gt) %.4f %.4f %.4f"%(diff_ori_l2, diff_est_l2, diff_gt_l2) )

        R_diff = torch.matmul(torch.inverse(Rs), rot_d['rotmats'])
        # print("R_diff", R_diff)
        cos_angle_diff = (torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)  - 1) / 2
        # print("cos_angle_diff", cos_angle_diff)
        cos_angle_diff = torch.clamp(cos_angle_diff, -1, 1)
        angle_diff = torch.acos(cos_angle_diff)
        angle_diff = angle_diff / np.pi * 180
        print("angle_diff (res, gt) %.4f %.4f"%(torch.abs(angle_diff).max().item(), torch.abs(rot_d['angles']).max().item() ) )

        t_diff = t_est - pts_d['t']
        t_diff_l2 = torch.norm(t_diff).item()
        t_l2 = torch.norm(pts_d['t']).item()
        t_1 = torch.norm(pts_d['t_1']).item()
        t_2 = torch.norm(pts_d['t_2']).item()
        print("t_diff (res, gt, t1, t2) %.4f %.4f %.4f %.4f"%(t_diff_l2, t_l2, t_1, t_2) )

        f_res.write("{:.4f} {:.4f}\n".format(torch.abs(angle_diff).max().item(), t_diff_l2))

        metric_dict['num'] += 1
        metric_dict['angle_sum'] += torch.abs(angle_diff).max().item()
        metric_dict['trans_sum'] += t_diff_l2
        angle_diff_90m = torch.abs(angle_diff).item()
        angle_diff_90p = 180-torch.abs(angle_diff).item()
        if angle_diff_90m < angle_diff_90p:
            metric_dict['angle180_sum'] += angle_diff_90m
            metric_dict['num_90-'] += 1
        else:
            metric_dict['angle180_sum'] += angle_diff_90p
            metric_dict['num_90+'] += 1        
        
        ### metric summary:
        angle_avg = metric_dict['angle_sum'] / metric_dict['num']
        angle180_avg = metric_dict['angle180_sum'] / metric_dict['num']
        trans_avg = metric_dict['trans_sum'] / metric_dict['num']
        lower90 = metric_dict['num_90-'] / metric_dict['num']
        higher90 = metric_dict['num_90+'] / metric_dict['num']
        print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(angle_avg, trans_avg, angle180_avg, lower90, higher90))

        print("----------------")


    if generate_mesh:
        t0 = time.time()
        out = generator.generate_mesh(data)
        time_dict['mesh'] = time.time() - t0

        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        time_dict.update(stats_dict)

        # Write output
        mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
        mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file

    if generate_pointcloud:
        t0 = time.time()
        pointcloud = generator.generate_pointcloud(data)
        time_dict['pcl'] = time.time() - t0
        pointcloud_out_file = os.path.join(
            pointcloud_dir, '%s.ply' % modelname)
        export_pointcloud(pointcloud, pointcloud_out_file)
        out_file_dict['pointcloud'] = pointcloud_out_file

    if cfg['generation']['copy_input']:
        # Save inputs
        if input_type == 'img':
            inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            visualize_data(inputs, 'img', inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'voxels':
            inputs_path = os.path.join(in_dir, '%s.off' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            voxel_mesh = VoxelGrid(inputs).to_mesh()
            voxel_mesh.export(inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'pointcloud':
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
            inputs = data['inputs'].squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in'] = inputs_path
            inputs_path_vis = os.path.join(in_dir, '%s_pclvis.jpg' % modelname)
            visualize_data(inputs, 'pointcloud', inputs_path_vis)
            out_file_dict['in_pclvis'] = inputs_path_vis

            inputs_path = os.path.join(in_dir, '%s_1_ori.ply' % modelname)
            # inputs = data['inputs'].squeeze(0).cpu().numpy()
            inputs = input_1.squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in_1_ori'] = inputs_path
            inputs_path_vis = os.path.join(in_dir, '%s_1_ori_pclvis.jpg' % modelname)
            visualize_data(inputs, 'pointcloud', inputs_path_vis)
            out_file_dict['in_1_ori_pclvis'] = inputs_path_vis
            
            inputs_path = os.path.join(in_dir, '%s_2_ori.ply' % modelname)
            # inputs = data['inputs'].squeeze(0).cpu().numpy()
            inputs = input_2.squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in_2_ori'] = inputs_path
            inputs_path_vis = os.path.join(in_dir, '%s_2_ori_pclvis.jpg' % modelname)
            visualize_data(inputs, 'pointcloud', inputs_path_vis)
            out_file_dict['in_2_ori_pclvis'] = inputs_path_vis

            inputs_path = os.path.join(in_dir, '%s_2_rot.ply' % modelname)
            # inputs = data['inputs'].squeeze(0).cpu().numpy()
            inputs = input_2_rot.squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in_2_rot'] = inputs_path
            inputs_path_vis = os.path.join(in_dir, '%s_2_rot_pclvis.jpg' % modelname)
            visualize_data(inputs, 'pointcloud', inputs_path_vis)
            out_file_dict['in_2_rot_pclvis'] = inputs_path_vis
            
            inputs_path = os.path.join(in_dir, '%s_1_rot_est.ply' % modelname)
            # inputs = data['inputs'].squeeze(0).cpu().numpy()
            inputs = input_1_rot_est.squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in_1_rot_est'] = inputs_path
            inputs_path_vis = os.path.join(in_dir, '%s_1_rot_est_pclvis.jpg' % modelname)
            visualize_data(inputs, 'pointcloud', inputs_path_vis)
            out_file_dict['in_1_rot_est_pclvis'] = inputs_path_vis
            ### add visualization here. Generate vis img using visualize_data() and save it in out/pointcloud/onet_vn/trained_286/input, then copy it to out/pointcloud/onet_vn/trained_286/vis below renaming it to sequenctial ids

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        # Save output files
        img_name = '%02d.off' % c_it
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir, '%02d_%04d_%s%s'
                                    % (c_it, it, k, ext))
            shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

### metric summary:
angle_avg = metric_dict['angle_sum'] / metric_dict['num']
angle180_avg = metric_dict['angle180_sum'] / metric_dict['num']
trans_avg = metric_dict['trans_sum'] / metric_dict['num']
lower90 = metric_dict['num_90-'] / metric_dict['num']
higher90 = metric_dict['num_90+'] / metric_dict['num']
print("=================================================")
print("angle_avg: ", angle_avg)
print("angle180_avg: ", angle180_avg)
print("trans_avg: ", trans_avg)
print("lower90: ", lower90)
print("higher90: ", higher90)
### final metrics needed:  mean angle diff, mean trans diff, mean angle diff to 180
f_res.close()
with open(os.path.join(generation_dir, "metrics.txt"), 'w') as f_met:
    f_met.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(angle_avg, trans_avg, angle180_avg, lower90, higher90))


# # Create pandas dataframe and save
# time_df = pd.DataFrame(time_dicts)
# time_df.set_index(['idx'], inplace=True)
# time_df.to_pickle(out_time_file)

# # Create pickle files  with main statistics
# time_df_class = time_df.groupby(by=['class name']).mean()
# time_df_class.to_pickle(out_time_file_class)

# # Print results
# time_df_class.loc['mean'] = time_df_class.mean()
# print('Timings [s]:')
# print(time_df_class)
