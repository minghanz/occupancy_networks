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

import numpy as np
try:
    from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
except Exception as e:
    print(e)
    print("pytorch 3d may not be installed. You may not be able to run register.py on this machine. ")

import se3, so3, invmat

class PointLK(torch.nn.Module):
    """IC LK on the SO(3) equivariant features of the point cloud. The Jacobian has simple analytic solution. """
    def __init__(self):
        super().__init__()
        # self.ptnet = ptnet
        self.inverse = invmat.InvMatrix.apply
        self.exp = so3.Exp # [B, 6] -> [B, 4, 4]
        self.transform = so3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        # w1 = delta
        # w2 = delta
        # w3 = delta
        # # v1 = delta
        # # v2 = delta
        # # v3 = delta
        # twist = torch.Tensor([w1, w2, w3])
        # self.dt = torch.nn.Parameter(twist.view(1, 3), requires_grad=learn_delta)

        # results
        self.last_err = None
        self.g_series = None # for debug purpose
        self.prev_r = None
        self.g = None # estimation result
        self.itr = 0

    @staticmethod
    def do_forward(net, p0, p1, maxiter=10, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        # a0 = torch.eye(3).view(1, 3, 3).expand(p0.size(0), 3, 3).to(p0) # [B, 4, 4]
        # a1 = torch.eye(3).view(1, 3, 3).expand(p1.size(0), 3, 3).to(p1) # [B, 4, 4]
        # if p0_zero_mean:
        #     p0_m = p0.mean(dim=1) # [B, N, 3] -> [B, 3]
        #     a0[:, 0:3, 3] = p0_m
        #     q0 = p0 - p0_m.unsqueeze(1)
        # else:
        # q0 = p0

        # if p1_zero_mean:
        #     #print(numpy.any(numpy.isnan(p1.numpy())))
        #     p1_m = p1.mean(dim=1) # [B, N, 3] -> [B, 3]
        #     a1[:, 0:3, 3] = -p1_m
        #     q1 = p1 - p1_m.unsqueeze(1)
        # else:
        # q1 = p1

        r = net(p0, p1, maxiter=maxiter, xtol=xtol)

        return r

    def forward(self, c0, c1, maxiter=10, xtol=1.0e-7, huber_delta=0):
        batch_size = c0.size(0)
        g0 = torch.eye(3).to(c0).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()
        res, g, itr = self.iclk(g0, c0, c1, maxiter, xtol, huber_delta)

        self.g = g
        self.itr = itr
        return res

    def iclk(self, g0, c0, c1, maxiter, xtol, huber_delta=0):
        ### goal: R * c0 = c1, R = exp(g), c0 = R^-1 * c1 = c1 + d(R^-1 c1) / dg * delta_g
        ### J = d(R^-1 c1) / dg, res = robust_norm(c1 - c0)
        ### delta_g = J^-1 * (-res)
        ### R_new = delta_R * R (c0 = delta_R * c0)

        batch_size = c0.size(0)
        # g0 = torch.eye(3).to(c0).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()

        g = g0
        self.g_series = torch.zeros(maxiter+1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        ### calculate Jacobian J and pseudo inverse J^-1= (JTJ)^-1 JT
        # approx. J by finite difference
        # dt = self.dt.to(c0).expand(batch_size, 3)
        # with torch.no_grad():
        J = self.approx_Jic(c0, c1)

        self.last_err = None
        itr = -1
        # try:
        # compute pinv(J) to solve J*x = -r
        # pinv = torch.pinverse(J)
        Jt = J.transpose(-1, -2)    # B*3*K
        JtJ = Jt.bmm(J)
        # logging.info("JtJ {}".format(JtJ))
        with torch.no_grad():
            lambdaI = max(torch.abs(JtJ).mean(), 1) * torch.eye(3, device=c0.device, dtype=c0.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        pinv = torch.inverse(JtJ + lambdaI).bmm(Jt)
        # pinv = torch.inverse(JtJ).bmm(Jt)

        # except RuntimeError as err:
        #     logging.info(err)
        #     # singular...?
        #     self.last_err = err
        #     #print(err)
        #     # Perhaps we can use MP-inverse, but,...
        #     # probably, self.dt is way too small...
        #     # f1 = self.ptnet(p1) # [B, N, 3] -> [B, K]
        #     # r = f1 - f0
        #     res = c1 - c0
        #     res = res.reshape(res.shape[0], -1)
        #     # self.ptnet.train(training)
        #     return res, g, itr


        ### iterative update
        itr = 0
        res = None
        for itr in range(maxiter):
            self.prev_r = res
            c = self.transform(g.unsqueeze(1), c0) # [B, 1, 3, 3] x [B, K, 3] -> [B, K, 3]
            # f = self.ptnet(p) # [B, N, 3] -> [B, K]
            res = c1 - c
            res = res.reshape(res.shape[0], -1)
            # res, J = self.approx_Jic(c0, c1)    # B*K, B*K*3
            # pinv = torch.pinverse(J)            # B*3*K
            res_abs = torch.abs(res)
            # logging.info("res mean {}, max {}, min {}, med {}".format(res_abs.mean().item(), res_abs.max().item(), res_abs.min().item(), res_abs.median().item()))

            if huber_delta > 0:
                W = torch.where(res_abs < huber_delta, torch.ones_like(res), huber_delta / res_abs)   # B*K
                Jt = J.transpose(-1, -2) * (W.unsqueeze(-2))    # B*3*K
                JtJ = Jt.bmm(J)
                # pinv = torch.inverse(JtJ).bmm(Jt)
                with torch.no_grad():
                    lambdaI = torch.abs(JtJ).mean() * torch.eye(3, device=c0.device, dtype=c0.dtype).unsqueeze(0).expand(batch_size, -1, -1)
                pinv = torch.inverse(JtJ + lambdaI).bmm(Jt)
                # pinv = torch.inverse(JtJ + 10 * torch.diag_embed(torch.diagonal(JtJ, dim1=-2, dim2=-1))).bmm(Jt)

            dx = -pinv.bmm(res.unsqueeze(-1)).view(batch_size, 3)

            # DEBUG.
            #norm_r = r.norm(p=2, dim=1)
            #print('itr,{},|r|,{}'.format(itr+1, ','.join(map(str, norm_r.data.tolist()))))

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0 # no update.
                break

            g = self.update(g, dx)
            self.g_series[itr+1] = g.clone()

        rep = len(range(itr, maxiter))
        self.g_series[(itr+1):] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        return res, g, (itr+1)

    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    def approx_Jic(self, c0, c1):
        dc1_dg = so3.mat(c1)
        dc1_dg = dc1_dg.reshape(dc1_dg.shape[0], -1, 3)
        return dc1_dg

    def approx_Jnic(self, c0, c1):       # non-IC version
        """c0, c1: B*K*3"""
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]



        dc0_dg = - so3.mat(c0)   # B*K*3*3
        res_d = torch.norm(c0 - c1, dim=-1, keepdim=True)   # B*K*1
        res_d2 = ((c0-c1)**2).sum(-1, keepdim=True)         # B*K*1

        huber_delta = 1e-3
        res_huber = torch.where(res_d < huber_delta, res_d**2, 2 * huber_delta*(res_d - 0.5 * huber_delta))     # B*K*1

        dhuber_dresd = torch.where(res_d < huber_delta, 2 * res_d, torch.ones_like(res_d) * 2 * huber_delta)    # B*K*1

        dresd_dc0 = (c0 - c1) / res_d   # B*K*3
        dresd2_dc0 = 2 * (c0 - c1)      # B*K*3
        
        dhuber_dg = torch.matmul((dhuber_dresd * dresd_dc0).unsqueeze(-2), dc0_dg).squeeze(-2)  # B*K*1*3 x B*K*3*3 -> B*K*1*3 -> B*K*3
        res_huber = res_huber.squeeze(-1)   # B*K

        return res_huber, dhuber_dg