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

from im2mesh.data.transforms import apply_rot

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
        logging.info('Warning: generator does not support mesh generation.')

    if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
        generate_pointcloud = False
        logging.info('Warning: generator does not support pointcloud generation.')
    record_worst_best = cfg['generation']['record_worst_best']
    
    output_regis_result = cfg['generation'].get('output_regis_result', False)

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
    logging.info("cfg['data']['pointcloud_n_val'] {}".format(cfg['data']['pointcloud_n_val']))
    logging.info("angle_avg, trans_avg, angle180_avg, lower90, higher90")
    metric_dict = dict()
    metric_dict['angle_sum'] = 0
    metric_dict['angle180_sum'] = 0
    metric_dict['trans_sum'] = 0
    metric_dict['num'] = 0
    metric_dict['num_90-'] = 0
    metric_dict['num_90+'] = 0
    metric_dict['rmse_sum'] = 0
    metric_dict['rmse_tp'] = 0

    metric_dict_lk = metric_dict.copy()

    if record_worst_best:
        worst_dict = dict()
        worst_n = 10
        worst_dict['loss'] = [0] * worst_n
        worst_dict['pcl_1'] = [None]*worst_n
        worst_dict['pcl_1_rot_est'] = [None]*worst_n
        worst_dict['pcl_1_rot_gt'] = [None]*worst_n
        worst_dict['pcl_2'] = [None]*worst_n
        worst_dict['category_name'] = [None]*worst_n

        worst90_dict = dict()
        worst90_n = 10
        worst90_dict['loss'] = [0] * worst90_n
        worst90_dict['pcl_1'] = [None]*worst90_n
        worst90_dict['pcl_1_rot_est'] = [None]*worst90_n
        worst90_dict['pcl_1_rot_gt'] = [None]*worst90_n
        worst90_dict['pcl_2'] = [None]*worst90_n
        worst90_dict['category_name'] = [None]*worst90_n
        worst90_dict['angle_diff'] = [None]*worst90_n

        worst45_dict = dict()
        worst45_n = 10
        worst45_dict['loss'] = [0] * worst45_n
        worst45_dict['pcl_1'] = [None]*worst45_n
        worst45_dict['pcl_1_rot_est'] = [None]*worst45_n
        worst45_dict['pcl_1_rot_gt'] = [None]*worst45_n
        worst45_dict['pcl_2'] = [None]*worst45_n
        worst45_dict['category_name'] = [None]*worst45_n
        worst45_dict['angle_diff'] = [None]*worst45_n

        best_dict = dict()
        best_n = 10
        best_dict['loss'] = [0] * best_n
        best_dict['pcl_1'] = [None]*best_n
        best_dict['pcl_1_rot_est'] = [None]*best_n
        best_dict['pcl_1_rot_gt'] = [None]*best_n
        best_dict['pcl_2'] = [None]*best_n
        best_dict['category_name'] = [None]*best_n

    lk_mode = cfg['test'].get('lk_mode', False)
    logging.info("lk_mode {}".format(lk_mode))
    if lk_mode:
        lk = PointLK()

    f_res = open(os.path.join(generation_dir, "results.txt"), "w")

    for it, data in enumerate(tqdm(test_loader)):

        # if it > 0:
        #     break

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)

        # Output folders
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
        in_dir = os.path.join(generation_dir, 'input')
        generation_vis_dir = os.path.join(generation_dir, 'vis', )
        output_regis_dir = os.path.join(generation_dir, 'regis_result')

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
            output_regis_dir = os.path.join(output_regis_dir, folder_name, modelname)   # keep the same structure as the input data folder

        # Create directories if necessary
        if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
            os.makedirs(generation_vis_dir)

        if generate_mesh and not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)

        if generate_pointcloud and not os.path.exists(pointcloud_dir):
            os.makedirs(pointcloud_dir)

        if not os.path.exists(in_dir):
            os.makedirs(in_dir)

        if output_regis_result and not os.path.exists(output_regis_dir):
            os.makedirs(output_regis_dir)
        
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
            out_1, out_2_rot = generator.generate_latent_conditioned(data)
            
            batch_size = out_1.shape[0]
            out_1 = out_1.reshape(batch_size, -1, 3)
            out_2_rot = out_2_rot.reshape(batch_size, -1, 3)

            input_1 = data['inputs']
            input_2_rot = data['inputs_2']
            R_gt = data['T21']
            rotdeg = data['T21.deg']

            out_1_rot = apply_rot(R_gt, out_1)

            R_est = solve_R(out_1, out_2_rot)
            out_1_rot_est = apply_rot(R_est, out_1)

            diff_ori = out_1 - out_2_rot
            diff_gt  = out_1_rot - out_2_rot
            diff_est = out_1_rot_est - out_2_rot

            input_1_rot = apply_rot(R_gt, input_1)

            # input_1_rot_64 = apply_rot(R_gt.to(torch.float64), input_1.to(torch.float64)).to(torch.float32)
            # input_1_rot_cpu = apply_rot(R_gt.cpu(), input_1.cpu())
            # input_1_rot_cpu_64 = apply_rot(R_gt.cpu().to(torch.float64), input_1.cpu().to(torch.float64)).to(torch.float32)
            # input_1_rot_np = np.matmul(R_gt.detach().cpu().numpy()[0], input_1.detach().cpu().numpy()[0].swapaxes(-1, -2))[None, ...].swapaxes(-1, -2)

            input_1_rot_est = apply_rot(R_est, input_1)
            input_2 = apply_rot(torch.inverse(R_gt), input_2_rot)
            

            # diff_pts_rmse_1 = torch.norm(input_1_rot_cpu.cuda() - input_1_rot, dim=2).mean()
            # diff_pts_rmse_2 = torch.norm(input_1_rot_cpu_64.cuda() - input_1_rot, dim=2).mean()
            # diff_pts_rmse_3 = torch.norm(input_1_rot_cpu.cuda() - input_1_rot_64, dim=2).mean()
            # diff_pts_rmse_4 = torch.norm(input_1_rot_cpu_64.cuda() - input_1_rot_64, dim=2).mean()
            # diff_pts_rmse_5 = torch.norm(torch.from_numpy(input_1_rot_np).cuda() - input_1_rot, dim=2).mean()
            # diff_pts_rmse_6 = torch.norm(torch.from_numpy(input_1_rot_np).cuda() - input_1_rot_64, dim=2).mean()
            # diff_pts_rmse_7 = torch.norm(torch.from_numpy(input_1_rot_np).cuda() - input_1_rot_cpu.cuda(), dim=2).mean()
            # diff_pts_rmse_8 = torch.norm(torch.from_numpy(input_1_rot_np).cuda() - input_1_rot_cpu_64.cuda(), dim=2).mean()
            # logging.info("diff_pts_rmse %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"%(diff_pts_rmse_1, diff_pts_rmse_2, diff_pts_rmse_3, diff_pts_rmse_4, diff_pts_rmse_5, diff_pts_rmse_6, diff_pts_rmse_7, diff_pts_rmse_8 ) )
            # logging.info(input_1_rot_cpu.cuda() - input_1_rot)

            diff_pts_rmse_est = torch.norm(input_1_rot_est - input_1_rot, dim=2).mean()
            diff_pts_rmse = torch.norm(input_2_rot - input_1_rot, dim=2).mean()
            # logging.info(input_2_rot - input_1_rot)
            logging.info("diff_pts_rmse %.4f %.4f "%(diff_pts_rmse, diff_pts_rmse_est ) )
            # logging.info("out data['T21'].dtype {} shape {} device {}".format(data['T21'].dtype, data['T21'].shape, data['T21'].device))
            # logging.info("out data['inputs'].dtype {} shape {} device {}".format(data['inputs'].dtype, data['inputs'].shape, data['inputs'].device))
            # logging.info("out input_1_rot.dtype {} shape {} device {}".format(input_1_rot.dtype, input_1_rot.shape, input_1_rot.device))
            # logging.info("out data['inputs_2'].dtype {} shape {} device {}".format(data['inputs_2'].dtype, data['inputs_2'].shape, data['inputs_2'].device))

            # diff_gt = diff_gt.cpu().detach()
            # diff_ori = diff_ori.cpu().detach()
            # diff_est = diff_est.cpu().detach()

            diff_gt_inf = torch.abs(diff_gt).max().item()
            diff_ori_inf = torch.abs(diff_ori).max().item()
            diff_est_inf = torch.abs(diff_est).max().item()

            diff_gt_l2 = torch.norm(diff_gt).item()     # == torch.sqrt((diff_gt**2).sum())
            diff_ori_l2 = torch.norm(diff_ori).item()
            diff_est_l2 = torch.norm(diff_est).item()

            ### previous: gt, ori, est
            if False:
            # if lk_mode:
                logging.info("feat_diff_Linf (ori, est, gt, lk) %.4f %.4f %.4f %.4f"%(diff_ori_inf, diff_est_inf, diff_gt_inf, diff_est_lk_inf) )
                logging.info("feat_diff_L2 (ori, est, gt, lk) %.4f %.4f %.4f %.4f"%(diff_ori_l2, diff_est_l2, diff_gt_l2, diff_est_lk_l2) )
            else:
                logging.info("feat_diff_Linf (ori, est, gt) %.4f %.4f %.4f"%(diff_ori_inf, diff_est_inf, diff_gt_inf) )
                logging.info("feat_diff_L2 (ori, est, gt) %.4f %.4f %.4f"%(diff_ori_l2, diff_est_l2, diff_gt_l2) )

            angle_diff = angle_diff_func(R_est, R_gt)
            logging.info("angle_diff (res, gt) %.4f %.4f"%(torch.abs(angle_diff).max().item(), torch.abs(rotdeg).max().item() ) )

            # if lk_mode:
            #     angle_diff_lk = angle_diff_func(Rs_lk, rot_d['rotmats'])
            #     logging.info("angle_diff_lk (res, gt) %.4f %.4f"%(torch.abs(angle_diff_lk).max().item(), torch.abs(rot_d['angles']).max().item() ) )


            f_res.write("{:.4f}\n".format(torch.abs(angle_diff).max().item()))
        
            update_metric_dict(metric_dict, angle_diff, 0, diff_pts_rmse, 'svd')
            # if lk_mode:
            #     update_metric_dict(metric_dict_lk, angle_diff_lk, t_diff_l2, diff_pts_rmse, 'lk')
        
            logging.info(str(category_id))
            ### update best and worst
            if record_worst_best:
                ds_cur = dict()
                ds_cur['pcl_1'] = input_1.squeeze(0).cpu().numpy()   # N*3
                ds_cur['pcl_1_rot_est'] = input_1_rot_est.squeeze(0).cpu().numpy()   # N*3
                ds_cur['pcl_1_rot_gt'] = input_1_rot.squeeze(0).cpu().numpy()   # N*3
                ds_cur['pcl_2'] = input_2_rot.squeeze(0).cpu().numpy()   # N*3
                ds_cur['category_name'] = str(category_id)
                loss_cur = angle_diff.item()
                update_worst(loss_cur, ds_cur, worst_dict)
                update_worst(loss_cur, ds_cur, best_dict, True)

                ds_cur['angle_diff'] = loss_cur
                update_worst(abs(loss_cur-90), ds_cur, worst90_dict, True)
                update_worst(min(abs(loss_cur-45), abs(loss_cur-135)), ds_cur, worst45_dict, True)

            logging.info("----------------")


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

        if output_regis_result:
            ### save point cloud pairs after registration and estimated rotation to file
            pcl_1_fname = cfg['data']['pointcloud_file_bench_1']
            pcl_1_fname = os.path.basename(pcl_1_fname)
            pcl_1_fname, ext = os.path.splitext(pcl_1_fname)
            pcl_1_fname = pcl_1_fname + '_est' + ext
            input_1_rot_np = input_1_rot[0].cpu().numpy()
            assert input_1_rot_np.ndim == 2, input_1_rot_np.shape
            np.save(os.path.join(output_regis_dir, pcl_1_fname), input_1_rot_np)

            T21_fname = cfg['data']['T21_file']
            T21_fname = os.path.basename(T21_fname)
            T21_fname, ext = os.path.splitext(T21_fname)
            T21_fname = T21_fname + '_est' + ext
            T21_est = R_est[0].cpu().numpy()
            angle_est = angle_of_R(R_est).cpu().numpy()
            assert angle_est.ndim == 1, angle_est.shape
            np.savez(os.path.join(output_regis_dir, T21_fname), T=T21_est, deg=angle_est)


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

    f_res.close()
    ### metric summary:
    summary_metric_dict(metric_dict, os.path.join(generation_dir, "metrics.txt"))
    if lk_mode:
        summary_metric_dict(metric_dict_lk, os.path.join(generation_dir, "metrics_lk.txt"))

    if record_worst_best:
        vis_dir = os.path.join(generation_dir, 'vis_results')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        logging.info("Saving worst cases to {}".format(vis_dir))
        for i in range(len(worst_dict['loss'])):
            color_1 = worst_dict['pcl_1_rot_gt'][i].mean(1)
            color_2 = worst_dict['pcl_2'][i].mean(1)
            inputs_path_vis = os.path.join(vis_dir, 'worst_%d_pcl_1.jpg' %i)
            visualize_data(worst_dict['pcl_1'][i], 'pointcloud', inputs_path_vis, info="src {}, {:.3f}".format(worst_dict['category_name'][i], worst_dict['loss'][i]), c1=color_1)
            inputs_path_vis = os.path.join(vis_dir, 'worst_%d_pcl_2.jpg' %i)
            visualize_data(worst_dict['pcl_2'][i], 'pointcloud', inputs_path_vis, info="tgt {}, {:.3f}".format(worst_dict['category_name'][i], worst_dict['loss'][i]), c1=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'worst_%d_pcl_1est_2.jpg' %i)
            visualize_data(worst_dict['pcl_1_rot_est'][i], 'pointcloud', inputs_path_vis, worst_dict['pcl_2'][i], info="est {}, {:.3f}".format(worst_dict['category_name'][i], worst_dict['loss'][i]), c1=color_1, c2=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'worst_%d_pcl_1gt_2.jpg' %i)
            visualize_data(worst_dict['pcl_1_rot_gt'][i], 'pointcloud', inputs_path_vis, worst_dict['pcl_2'][i], info="gt {}, {:.3f}".format(worst_dict['category_name'][i], worst_dict['loss'][i]), c1=color_1, c2=color_2)

        logging.info("Saving best cases to {}".format(vis_dir))
        for i in range(len(best_dict['loss'])):
            color_1 = best_dict['pcl_1_rot_gt'][i].mean(1)
            color_2 = best_dict['pcl_2'][i].mean(1)
            inputs_path_vis = os.path.join(vis_dir, 'best_%d_pcl_1.jpg' %i)
            visualize_data(best_dict['pcl_1'][i], 'pointcloud', inputs_path_vis, info="src {}, {:.3f}".format(best_dict['category_name'][i], best_dict['loss'][i]), c1=color_1)
            inputs_path_vis = os.path.join(vis_dir, 'best_%d_pcl_2.jpg' %i)
            visualize_data(best_dict['pcl_2'][i], 'pointcloud', inputs_path_vis, info="tgt {}, {:.3f}".format(best_dict['category_name'][i], best_dict['loss'][i]), c1=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'best_%d_pcl_1est_2.jpg' %i)
            visualize_data(best_dict['pcl_1_rot_est'][i], 'pointcloud', inputs_path_vis, best_dict['pcl_2'][i], info="est {}, {:.3f}".format(best_dict['category_name'][i], best_dict['loss'][i]), c1=color_1, c2=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'best_%d_pcl_1gt_2.jpg' %i)
            visualize_data(best_dict['pcl_1_rot_gt'][i], 'pointcloud', inputs_path_vis, best_dict['pcl_2'][i], info="gt {}, {:.3f}".format(best_dict['category_name'][i], best_dict['loss'][i]), c1=color_1, c2=color_2)
        
        logging.info("Saving worst90 cases to {}".format(vis_dir))
        for i in range(len(worst90_dict['loss'])):
            color_1 = worst90_dict['pcl_1_rot_gt'][i].mean(1)
            color_2 = worst90_dict['pcl_2'][i].mean(1)
            inputs_path_vis = os.path.join(vis_dir, 'worst90_%d_pcl_1.jpg' %i)
            visualize_data(worst90_dict['pcl_1'][i], 'pointcloud', inputs_path_vis, info="src {}, {:.3f}".format(worst90_dict['category_name'][i], worst90_dict['angle_diff'][i]), c1=color_1)
            inputs_path_vis = os.path.join(vis_dir, 'worst90_%d_pcl_2.jpg' %i)
            visualize_data(worst90_dict['pcl_2'][i], 'pointcloud', inputs_path_vis, info="tgt {}, {:.3f}".format(worst90_dict['category_name'][i], worst90_dict['angle_diff'][i]), c1=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'worst90_%d_pcl_1est_2.jpg' %i)
            visualize_data(worst90_dict['pcl_1_rot_est'][i], 'pointcloud', inputs_path_vis, worst90_dict['pcl_2'][i], info="est {}, {:.3f}".format(worst90_dict['category_name'][i], worst90_dict['angle_diff'][i]), c1=color_1, c2=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'worst90_%d_pcl_1gt_2.jpg' %i)
            visualize_data(worst90_dict['pcl_1_rot_gt'][i], 'pointcloud', inputs_path_vis, worst90_dict['pcl_2'][i], info="gt {}, {:.3f}".format(worst90_dict['category_name'][i], worst90_dict['angle_diff'][i]), c1=color_1, c2=color_2)

        logging.info("Saving worst45 cases to {}".format(vis_dir))
        for i in range(len(worst45_dict['loss'])):
            color_1 = worst45_dict['pcl_1_rot_gt'][i].mean(1)
            color_2 = worst45_dict['pcl_2'][i].mean(1)
            inputs_path_vis = os.path.join(vis_dir, 'worst45_%d_pcl_1.jpg' %i)
            visualize_data(worst45_dict['pcl_1'][i], 'pointcloud', inputs_path_vis, info="src {}, {:.3f}".format(worst45_dict['category_name'][i], worst45_dict['angle_diff'][i]), c1=color_1)
            inputs_path_vis = os.path.join(vis_dir, 'worst45_%d_pcl_2.jpg' %i)
            visualize_data(worst45_dict['pcl_2'][i], 'pointcloud', inputs_path_vis, info="tgt {}, {:.3f}".format(worst45_dict['category_name'][i], worst45_dict['angle_diff'][i]), c1=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'worst45_%d_pcl_1est_2.jpg' %i)
            visualize_data(worst45_dict['pcl_1_rot_est'][i], 'pointcloud', inputs_path_vis, worst45_dict['pcl_2'][i], info="est {}, {:.3f}".format(worst45_dict['category_name'][i], worst45_dict['angle_diff'][i]), c1=color_1, c2=color_2)
            inputs_path_vis = os.path.join(vis_dir, 'worst45_%d_pcl_1gt_2.jpg' %i)
            visualize_data(worst45_dict['pcl_1_rot_gt'][i], 'pointcloud', inputs_path_vis, worst45_dict['pcl_2'][i], info="gt {}, {:.3f}".format(worst45_dict['category_name'][i], worst45_dict['angle_diff'][i]), c1=color_1, c2=color_2)


    # # Create pandas dataframe and save
    # time_df = pd.DataFrame(time_dicts)
    # time_df.set_index(['idx'], inplace=True)
    # time_df.to_pickle(out_time_file)

    # # Create pickle files  with main statistics
    # time_df_class = time_df.groupby(by=['class name']).mean()
    # time_df_class.to_pickle(out_time_file_class)

    # # logging.info results
    # time_df_class.loc['mean'] = time_df_class.mean()
    # logging.info('Timings [s]:')
    # logging.info(time_df_class)
