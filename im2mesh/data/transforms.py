import numpy as np

import so3, se3
import torch
import random

def apply_transformation(T, pts):
    '''rotmat: ?*4*4, pts: ?*N*3'''
    assert pts.shape[-1] == 3, pts.shape
    if pts.ndim == 1:
        pts = pts.unsqueeze(0)
    assert pts.ndim == T.ndim, "{} {}".format(pts.shape, T.shape)

    T = T.to(pts)
    pts_trans = se3.transform(T, pts.transpose(-1, -2)).transpose(-1, -2)
    return pts_trans

# def apply_rot(rotmat, pts):
#     '''rotmat: ?*3*3, pts: N*3'''
#     assert pts.ndim == 2 and pts.shape[1] == 3, pts.shape
#     if rotmat.ndim == 2:
#         rotmat = rotmat.unsqueeze(0)
#     else:
#         assert rotmat.shape[0] == pts.shape[0]
#     rotmat = rotmat.to(pts)
#     pts_rot = so3.transform(rotmat, pts)   # [1 or N,3,3] x [N,3] -> [N,3]
#     return pts_rot

def apply_rot(rotmat, pts):
    '''rotmat: ?*3*3, pts: ?*N*3'''
    assert pts.shape[-1] == 3, pts.shape
    if pts.ndim == 1:
        pts = pts.unsqueeze(0)
    assert pts.ndim == rotmat.ndim, "{} {}".format(pts.shape, rotmat.shape)

    rotmat = rotmat.to(pts)
    pts_rot = so3.transform(rotmat, pts.transpose(-1, -2)).transpose(-1, -2)
    return pts_rot

def gen_randrot(mag_max=None, mag_random=True):
    # tensor: [N, 3]
    mag_max = 180 if mag_max is None else mag_max
    amp = torch.rand(1) if mag_random else 1.0
    deg = amp * mag_max
    w = torch.randn(1, 3)
    w = w / w.norm(p=2, dim=1, keepdim=True) * deg * np.pi / 180

    g = so3.exp(w)      # [1, 3, 3]
    g = g.squeeze(0)    # [3, 3]
    return g, deg

    # g = so3.exp(w).to(tensor)  # [1, 3, 3]
    # p1 = so3.transform(g, tensor)  # [1, 3, 3] x [N, 3] -> [N, 3]
    # return p1

def totensor_inplace(data):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = torch.from_numpy(value)
    return data

### Transforms for DualDataset
class CentralizePairBatchIP(object):
    '''In-place centralization transform for a batch of PairedDataset data'''
    def __init__(self) -> None:
        super().__init__()
        # self.device = device

    def __call__(self, data):
        # device = self.device
        inputs = data['inputs']
        inputs_2 = data['inputs_2']
        inputs_mean = inputs.mean(dim=1, keepdim=True)
        data['inputs'] = inputs - inputs_mean
        inputs_2_mean = inputs_2.mean(dim=1, keepdim=True)
        data['inputs_2'] = inputs_2 - inputs_2_mean
        
        if 'points' in data:
            data['points'] = data['points'] - inputs_mean
            data['points_2'] = data['points_2'] - inputs_2_mean
            
        return

class RotatePairBatchIP(object):
    def __init__(self) -> None:
        super().__init__()
        # self.device = device

    def __call__(self, data):
        # device = self.device
        
        data['inputs_2'] = apply_rot(data['T21'], data['inputs_2'])
        if 'points' in data:
            data['points_2'] = apply_rot(data['T21'], data['points_2'])
        return

def noise_pts(pts, stddev):
    noise = stddev * torch.randn(*pts.shape, dtype=pts.dtype, device=pts.device)
    pts = pts + noise
    return pts

class NoisePairBatchIP(object):
    def __init__(self, stddev, device=None) -> None:
        super().__init__()
        self.stddev = stddev
        self.device = device

    def __call__(self, data):
        device = self.device if self.device is not None else data['inputs'].device

        inputs = data.get('inputs').to(device)
        inputs_2 = data.get('inputs_2', inputs.clone()).to(device)

        inputs = noise_pts(inputs, self.stddev)
        inputs_2 = noise_pts(inputs_2, self.stddev)
        
        data['inputs'] = inputs
        data['inputs_2'] = inputs_2
        return

def subsample(pts, n):
    n_pts_in = pts.shape[1]

    if n <= 0:
        return pts
    elif n < n_pts_in:
        idx_sample = torch.randperm(n_pts_in)[:n]
        pts = pts[:, idx_sample]
        return pts
    elif n == n_pts_in:
        return pts
    else:
        raise ValueError("n=%d is more than the size of the point cloud %d"%(n, n_pts_in))

class SubSamplePairBatchIP(object):
    '''In-place subsampling transform for a batch of PairedDataset data'''
    def __init__(self, n1, n2_min, n2_max, device) -> None:
        super().__init__()
        self.n1 = n1
        self.n2_min = n2_min
        self.n2_max = n2_max
        
        self.device = device
     
    def __call__(self, data):
        device = self.device

        inputs = data.get('inputs')
        inputs_2 = data.get('inputs_2', inputs.clone())
        
        assert inputs.ndim == 3 and inputs.shape[1] == inputs_2.shape[1], "{}, {}".format(inputs.shape, inputs_2.shape)
        inputs = subsample(inputs, self.n1)

        # if 'inputs_2' not in data:
        #     n2 = random.randint(self.n2_min, self.n2_max)
        #     inputs_2 = subsample(inputs_2, n2)
        # else:
        #     inputs_2 = subsample(inputs_2, self.n1) # for 7scenes, avoid too few points
        n2 = random.randint(self.n2_min, self.n2_max)
        inputs_2 = subsample(inputs_2, n2)

        data['inputs'] = inputs.to(device)
        data['inputs_2'] = inputs_2.to(device)
        return

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        # indices = np.random.randint(points.shape[0], size=self.N)
        indices = np.random.permutation(points.shape[0])[:self.N]

        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            # idx = np.random.randint(points.shape[0], size=self.N)
            idx = np.random.permutation(points.shape[0])[:self.N]
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            # idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            # idx1 = np.random.randint(points1.shape[0], size=Nt_in)
            idx0 = np.random.permutation(points0.shape[0])[:Nt_out]
            idx1 = np.random.permutation(points1.shape[0])[:Nt_in]

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out
