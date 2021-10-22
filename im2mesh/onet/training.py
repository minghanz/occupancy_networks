import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist, sub
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer

from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations, axis_angle_to_matrix
import numpy as np
import random
from geometry import PointLK
import logging

from im2mesh.data.transforms import SubSamplePairBatchIP, CentralizePairBatchIP, RotatePairBatchIP

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item(), dict()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        # batch_size = data['points'].size(0)
        batch_size = data['inputs'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        # logging.info('inputs.shape {}'.format(inputs.shape))

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.model.decode(p, z, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss

def mat2angle(rotmat):
    cos_angle_diff = (torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(-1)  - 1) / 2
    # print("cos_angle_diff", cos_angle_diff)
    cos_angle_diff = torch.clamp(cos_angle_diff, -1, 1)
    angles = torch.acos(cos_angle_diff)
    return angles

def ang_mse_loss(A):
    I = torch.eye(3).to(A).view(1, 3, 3).expand(A.size(0), 3, 3)
    return torch.nn.functional.mse_loss(A, I, size_average=True) * 9

def ang_cos_loss(rotmat):
    diags = torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(-1)
    return -diags
    # cos_angle_diff = (torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(-1)  - 1) / 2
    # return - cos_angle_diff


# def gen_random_rot(inputs, rotate, device):
#     if rotate == -1:
#         ### use random rotations
#         rotmats = random_rotations(inputs.shape[0], dtype=inputs.dtype, device=device)
#         trot = Rotate(rotmats, device=device)
#         angles = mat2angle(rotmats)
#         angles = angles / np.pi * 180

#     else:
#         ### use random-axis-angle rotations
#         axis = (torch.rand(inputs.shape[0], 3, dtype=inputs.dtype) - 0.5)
#         axis = axis / torch.norm(axis, dim=1, keepdim=True)   # B*3
#         angles = torch.rand(inputs.shape[0]) * rotate
#         angles_rad = angles * np.pi / 180
#         axis_angle = axis * angles_rad.unsqueeze(1)
#         rotmats = axis_angle_to_matrix(axis_angle)
#         rotmats = rotmats.to(device=device)
#         trot = Rotate(rotmats, device=device)
#     out = dict(op=trot, mat=rotmats, angle_deg=angles)

#     return out

# class RotateDual(object):
#     def __init__(self, rotate, device) -> None:
#         super().__init__()
#         self.rotate = rotate
#         self.device = device

#     def __call__(self, data):
#         device = self.device

#         inputs_2 = data.get('inputs_2', data.get('inputs')).to(device)

#         d_rot = gen_random_rot(inputs_2, self.rotate, device)
#         trot = d_rot['op']

#         inputs_rot = trot.transform_points(inputs_2)
#         data['inputs_2'] = inputs_rot

#         if 'points' in data:
#             points_2 = data.get('points_2', data.get('points')).to(device)
#             points_rot = trot.transform_points(points_2)
#             data['points_2'] = points_rot

#         return d_rot

# class ShiftDual(object):
#     def __init__(self, shift_max, shift_sep, device) -> None:
#         super().__init__()
#         self.shift_max = shift_max
#         self.device = device
#         self.shift_sep = shift_sep

#     def __call__(self, data):
#         device = self.device

#         inputs = data.get('inputs').to(device)
#         inputs_2 = data.get('inputs_2', inputs.clone()).to(device)

#         shift = self.shift_max * (torch.rand(inputs.shape[0], 1, inputs.shape[2]) - 0.5) * 2
#         shift = shift.to(device)
#         inputs = inputs + shift
#         if self.shift_sep > 0:
#             shift_2 = self.shift_sep * torch.randn(inputs_2.shape[0], 1, inputs_2.shape[2])
#             shift_2 = shift_2.to(device) + shift
#             inputs_2 = inputs_2 + shift_2
#         else:
#             inputs_2 = inputs_2 + shift
#         data['inputs'] = inputs
#         data['inputs_2'] = inputs_2

#         if 'points' in data:
#             points = data.get('points').to(device)
#             points_2 = data.get('points_2', points.clone()).to(device)
#             points = points + shift
#             if self.shift_sep:
#                 points_2 = points_2 + shift_2
#             else:
#                 points_2 = points_2 + shift
#             data['points'] = points
#             data['points_2'] = points_2

#         return

def solve_R(f1, f2):
    """f1 and f2: (b*)m*3
    f2 * R -> f1
    """
    batch_size = f1.shape[0]

    S = torch.matmul(f1.transpose(-1, -2), f2)  #B*3*3
    try:
        U, sigma, V = torch.svd(S)
    except:
        print("adding noise to avoid divergence in torch.svd")
        noise = torch.diag_embed(torch.randn(batch_size, 3, device=S.device) * 1e-4)
        try:
            U, sigma, V = torch.svd(S + noise)
        except Exception as e:
            print(S)
            print("nan?", torch.any(torch.isnan(S)))
            raise ValueError(e)
        # print("S", S)
        # print("noise", noise)
        # U, sigma, V = torch.svd(S + 1e-4*S.mean()*torch.eye(S.shape[1], device=S.device).unsqueeze(0))
        # U, sigma, V = torch.svd(S + 1e-4*S.mean()*torch.rand(S.shape, device=S.device))
    R = torch.matmul(V, U.transpose(-1, -2))
    det = torch.det(R)
    # print(R)
    diag_1 = torch.tensor([1, 1, 0], device=R.device, dtype=R.dtype)
    diag_1 = diag_1.unsqueeze(0).expand(batch_size, -1)                 # B*3
    diag_2 = torch.tensor([0, 0, 1], device=R.device, dtype=R.dtype)
    diag_2 = diag_2.unsqueeze(0).expand(batch_size, -1)
    det = det.reshape(-1, 1)
    
    det_mat = torch.diag_embed(diag_1 + diag_2 * det) # B*3*3

    R = torch.matmul(V, torch.matmul(det_mat, U.transpose(-1, -2)))
    # print("det", det)
    
    return R

class DualTrainer(Trainer):
    def __init__(self, rotate=0, noise_std=0, shift_max=0, n1=0, n2_min=0, n2_max=0, angloss_w=1, closs_w=0, lk_mode=False, occloss_w=1, cos_loss=False, cos_mse=False, lk_supp=False, shift_sep=0, centralize=False, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.rotate = rotate
        # self.noise_std = noise_std
        # self.shift_max = shift_max
        # self.shift_sep = shift_sep
        self.angloss_w = angloss_w
        self.closs_w = closs_w
        self.lk_mode = lk_mode
        self.occloss_w = occloss_w
        self.cos_loss = cos_loss
        self.cos_mse = cos_mse
        self.lk_supp = lk_supp

        self.centralize = centralize

        self.sub_op = SubSamplePairBatchIP(n1, n2_min, n2_max, self.device)
        # self.noise_op = NoisePairBatchIP(noise_std, self.device)
        # self.shift_op = ShiftDual(shift_max, shift_sep, self.device)
        self.rotate_op = RotatePairBatchIP()

        self.ctr_op = CentralizePairBatchIP()

        if self.lk_mode or self.lk_supp:
            self.lk = PointLK()

    def compute_loss_single(self, data, idx):
        
        device = self.device
        dont_decode = 'points' not in data
        if not dont_decode:
            occ = data.get('points.occ').to(device)
        if idx == 1:
            inputs = data.get('inputs').to(device)
            if not dont_decode:
                p = data.get('points').to(device)
        else:
            assert idx == 2
            inputs = data.get('inputs_2').to(device)
            if not dont_decode:
                p = data.get('points_2').to(device)

        kwargs = {}

        c = self.model.encode_inputs(inputs)

        # q_z = self.model.infer_z(p, occ, c, **kwargs)
        # z = q_z.rsample()
        # logging.info("sampled z: {} {}".format(z.shape, z))

        # # KL-divergence
        # kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        # loss = kl.mean()
        # # print("kl", kl)# 0

        z = None

        if not dont_decode:
            # General points
            logits = self.model.decode(p, z, c, **kwargs).logits
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ, reduction='none')
            loss = loss_i.sum(-1).mean() # + loss

            # print("loss_i", loss_i)
        else:
            loss = torch.tensor(0, device=device, dtype=inputs.dtype)

        c = c.reshape(c.shape[0], -1, 3)
        return loss, c

    def compute_loss(self, data):
        device = self.device
        
        # inputs = data.get('inputs').to(device)
        # points = data.get('points').to(device)
        # kwargs = {}

        # input_max = torch.max(torch.abs(inputs))
        # norm_max = torch.max(torch.norm(inputs, dim=-1))
        # print("max inf, norm", input_max, norm_max)

        ### subsample -> shift -> noise -> rotate
        self.sub_op(data)
        # logging.info("inputs.shape {}, inputs_2.shape {}".format(data['inputs'].shape, data['inputs_2'].shape))
        # self.shift_op(data)
        # self.noise_op(data)
        # d_rot = self.rotate_op(data)
        self.rotate_op(data)
        if self.centralize:
            self.ctr_op(data)

        loss_1, c_1 = self.compute_loss_single(data, 1)
        loss_2, c_2 = self.compute_loss_single(data, 2)
        loss = loss_1 + loss_2

        # if self.lk_mode:
        #     r = self.lk(c_2, c_1) # , huber_delta=0.01
        #     R_est = self.lk.g
        # else:
        #     R_est = solve_R(c_2, c_1)   # R_est * c2(3*n) = c1(3*n), p1T(n*3) * R_gt = p2T (n*3)
        #     if self.lk_supp:
        #         R_gt_supp = torch.matmul(d_rot['mat'], R_est.detach().transpose(-2, -1))
        #         c2_r = torch.matmul(R_est.detach(), c_2.transpose(-2, -1)).transpose(-2, -1)
        #         r = self.lk(c2_r, c_1, huber_delta=0.01)
        #         R_est_supp = self.lk.g
        #         R_supp_res = torch.matmul(R_gt_supp.transpose(-2,-1), R_est_supp)
        #         loss_angle_mse_supp = ang_mse_loss(R_supp_res)
        #         # R_est_total = torch.matmul(R_est_supp, R_est.detach())
        #         # R_res_total = torch.matmul(d_rot['mat'].transpose(-2,-1), R_est_total)
        #         # loss_angle_mse_total = ang_mse_loss(R_res_total)

        # R_gt = d_rot['mat']

        if self.lk_mode:
            r = self.lk(c_1, c_2) # , huber_delta=0.01
            R_est = self.lk.g
        else:
            R_est = solve_R(c_1, c_2)

        R_gt = data['T21']

        # loss_angle = mat2angle(torch.matmul(R_est.transpose(-2,-1), R_gt))
        R_res = torch.matmul(R_gt.transpose(-2,-1), R_est)
        loss_angle = mat2angle(R_res)
        loss_angle_mean = loss_angle.mean()

        loss_angle_mse = ang_mse_loss(R_res)
        loss_cos = ang_cos_loss(R_res)
        loss_cos_mean = loss_cos.mean()
        loss_cos_mse = ((loss_cos + 3)**2).mean()
        if self.angloss_w > 0:
            if self.occloss_w == 0:
                # loss = loss_angle_mean * self.angloss_w
                if self.cos_loss:
                    if self.cos_mse:
                        loss = loss_cos_mse * self.angloss_w
                    else:
                        loss = loss_cos_mean * self.angloss_w
                else:
                    loss = loss_angle_mse * self.angloss_w
            else:
                # loss = loss * self.occloss_w + loss_angle_mean * self.angloss_w
                if self.cos_loss:
                    if self.cos_mse:
                        loss = loss * self.occloss_w + loss_cos_mse * self.angloss_w
                    else:
                        loss = loss * self.occloss_w + loss_cos_mean * self.angloss_w
                else:
                    loss = loss * self.occloss_w + loss_angle_mse * self.angloss_w

            # if self.lk_supp:
            #     loss = loss + loss_angle_mse_supp * self.angloss_w

        # trot_est = Rotate(R_est)
        # c_2_op = d_rot['op'].transform_points(c_1)

        # c_2_err = c_2_op - c_2
        # with torch.no_grad():
        #     print("c_1", torch.abs(c_1).mean())
        #     print("c_2", torch.abs(c_2).mean())
        #     print("c_2_op-c_2", torch.abs(c_2_err).mean())

        # loss_c = torch.abs(c_2_err).sum(-1).mean()
        # if self.closs_w > 0:
        #     loss = loss + loss_c * self.closs_w

        d_loss = dict(loss_1=loss_1.item(), loss_2=loss_2.item(), loss_ang=loss_angle_mean.item(), loss_ang_mse=loss_angle_mse.item(), loss_cos=loss_cos_mean.item(), loss_cos_mse=loss_cos_mse.item()) # , loss_c=loss_c.item()

        # if self.lk_supp:
        #     d_loss['loss_angle_mse_supp'] = loss_angle_mse_supp.item()

        return loss, d_loss
    
    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss, d_loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item(), d_loss