import torch
# import torch.distributions as dist

### https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
import logging

import numpy as np

def solve_R(f1, f2):
    """f1 and f2: (b*)m*3
    only work for batch_size=1
    """
    S = torch.matmul(f1.transpose(-1, -2), f2)  # 3*3
    U, sigma, V = torch.svd(S)
    R = torch.matmul(V, U.transpose(-1, -2))
    det = torch.det(R)
    # logging.info(R)
    diag_1 = torch.tensor([1, 1, 0], device=R.device, dtype=R.dtype)
    diag_2 = torch.tensor([0, 0, 1], device=R.device, dtype=R.dtype)
    det_mat = torch.diag(diag_1 + diag_2 * det)

    # det_mat = torch.eye(3, device=R.device, dtype=R.dtype)
    # det_mat[2, 2] = det

    det_mat = det_mat.unsqueeze(0)
    # logging.info(det_mat)
    R = torch.matmul(V, torch.matmul(det_mat, U.transpose(-1, -2)))
    logging.info(det)
    # logging.info(V.shape)
    
    return R

def angle_diff_func(R1, R2):
    R_diff = torch.matmul(torch.inverse(R1), R2)
    # logging.info("R_diff", R_diff)
    cos_angle_diff = (torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)  - 1) / 2
    # logging.info("cos_angle_diff", cos_angle_diff)
    cos_angle_diff = torch.clamp(cos_angle_diff, -1, 1)
    angle_diff = torch.acos(cos_angle_diff)
    angle_diff = angle_diff / np.pi * 180
    return angle_diff

def summary_metric_dict(metric_dict, path_out):
    ### metric summary:
    angle_avg = metric_dict['angle_sum'] / metric_dict['num']
    angle180_avg = metric_dict['angle180_sum'] / metric_dict['num']
    trans_avg = metric_dict['trans_sum'] / metric_dict['num']
    lower90 = metric_dict['num_90-'] / metric_dict['num']
    higher90 = metric_dict['num_90+'] / metric_dict['num']
    rmse_avg = metric_dict['rmse_sum'] / metric_dict['num']
    rmse_tp = metric_dict['rmse_tp'] / metric_dict['num']
    logging.info("=================================================")
    logging.info("angle_avg: {}".format(angle_avg))
    logging.info("angle180_avg: {}".format(angle180_avg))
    logging.info("trans_avg: {}".format(trans_avg))
    logging.info("lower90: {}".format(lower90))
    logging.info("higher90: {}".format(higher90))
    logging.info("rmse_avg: {}".format(rmse_avg))
    logging.info("rmse_tp: {}".format(rmse_tp))
    ### final metrics needed:  mean angle diff, mean trans diff, mean angle diff to 180
    if path_out is not None:
        with open(path_out, 'w') as f_met:
            f_met.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(angle_avg, trans_avg, angle180_avg, lower90, higher90, rmse_avg, rmse_tp))
    return

def update_metric_dict(metric_dict, angle_diff, t_diff_l2, rmse_pts, name):
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
    
    metric_dict['rmse_sum'] += rmse_pts.item()
    if rmse_pts <= 0.2: 
        metric_dict['rmse_tp'] += 1

    ### metric summary:
    angle_avg = metric_dict['angle_sum'] / metric_dict['num']
    angle180_avg = metric_dict['angle180_sum'] / metric_dict['num']
    trans_avg = metric_dict['trans_sum'] / metric_dict['num']
    lower90 = metric_dict['num_90-'] / metric_dict['num']
    higher90 = metric_dict['num_90+'] / metric_dict['num']
    rmse_avg = metric_dict['rmse_sum'] / metric_dict['num']
    rmse_tp = metric_dict['rmse_tp'] / metric_dict['num']
    logging.info("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(name, angle_avg, trans_avg, angle180_avg, lower90, higher90, rmse_avg, rmse_tp))

def update_worst(loss_in, ds_cur, ds_records, best_mode=False):
    i_replace = -1
    for i, worst_l in enumerate(ds_records['loss']):
        if best_mode:
            condition = worst_l == 0 or loss_in <= worst_l
        else:
            condition = loss_in >= worst_l
        if condition:
            i_replace = i
        else:
            break

    if i_replace > 0:
        for i in range(0, i_replace):
            for key in ds_records:
                ds_records[key][i] = ds_records[key][i+1]

    if i_replace >= 0:
        for key in ds_records:
            if key == 'loss':
                ds_records['loss'][i_replace] = loss_in
            else:
                ds_records[key][i_replace] = ds_cur[key]
    return
