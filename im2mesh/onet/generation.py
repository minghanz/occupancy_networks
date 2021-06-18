import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time

from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations, axis_angle_to_matrix

class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None,
                 preprocessor=None, 
                 rotate=-1, 
                 noise=0, 
                 centralize=False):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

        self.rotate = rotate
        self.noise = noise      # noise only effective when not sampling different points
        self.centralize = centralize

    def generate_latent_conditioned(self, data):
        self.model.eval()
        device = self.device
        stats_dict = {}

        # print("data.keys()", data.keys())   # ['points', 'points.occ', 'points_iou', 'points_iou.occ', 'voxels', 'inputs', 'inputs.normals', 'idx']

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        input_max = torch.max(torch.abs(inputs))
        norm_max = torch.max(torch.norm(inputs, dim=-1))
        print("max inf, norm", input_max, norm_max)

        if self.rotate == -1:
            ### use random rotations
            rotmats = random_rotations(inputs.shape[0], dtype=inputs.dtype)
            trot = Rotate(rotmats)

            cos_angle_diff = (torch.diagonal(rotmats, dim1=-2, dim2=-1).sum(-1)  - 1) / 2
            # print("cos_angle_diff", cos_angle_diff)
            cos_angle_diff = torch.clamp(cos_angle_diff, -1, 1)
            angles = torch.acos(cos_angle_diff)
            angles = angles / np.pi * 180

        else:
            ### use random-axis-angle rotations
            axis = (torch.rand(inputs.shape[0], 3, dtype=inputs.dtype) - 0.5)
            axis = axis / torch.norm(axis, dim=1)   # B*3
            angles = torch.rand(inputs.shape[0]) * self.rotate
            angles_rad = angles * np.pi / 180
            axis_angle = axis * angles_rad.unsqueeze(1)
            rotmats = axis_angle_to_matrix(axis_angle)
            trot = Rotate(rotmats)

        # ### use z-axis rotations
        # # print(inputs.shape) # 1*300*3
        # angles = torch.rand(inputs.shape[0])*360
        # trot = RotateAxisAngle(angle=angles, axis="Z", degrees=True)
        # rotmats = trot.get_matrix()[:, :3, :3]

        rot_mat_for_t = random_rotations(inputs.shape[0], dtype=inputs.dtype)       # B*3*3
        t_mag = np.random.random(inputs.shape[0])                                   # B
        t_0 = np.stack([t_mag, np.zeros_like(t_mag), np.zeros_like(t_mag)], axis=1) # B*3
        t_0 = torch.tensor(t_0, dtype=inputs.dtype).unsqueeze(2)                    # B*3*1
        t = torch.matmul(rot_mat_for_t, t_0).transpose(-1, -2)                      # B*1*3
        t = t.to(device)

        rot_d = {}
        rot_d['angles'] = angles
        rot_d['trot'] = trot
        rot_d['rotmats'] = rotmats
        rot_d['t'] = t

        ### create a rotated copy:
        n_points_in = inputs.shape[1]
        if n_points_in == 1024:
            idx_sample = torch.randperm(n_points_in)[:1024]
            idx_sample_2 = torch.randperm(n_points_in)[:102]
            inputs_2 = inputs
            inputs = inputs[:, idx_sample]
            inputs_2 = inputs_2[:, idx_sample_2]
        elif n_points_in > 1000:
            idx_sample = torch.randperm(n_points_in)[:1000]
            idx_sample_2 = torch.randperm(n_points_in)[:1000]
            inputs_2 = inputs
            inputs = inputs[:, idx_sample]
            inputs_2 = inputs_2[:, idx_sample_2]
        else:
            noise = torch.randn_like(inputs) * self.noise
            inputs_2 = inputs + noise

        inputs_rot = trot.transform_points(inputs_2.cpu())
        inputs_rot = inputs_rot.to(device=inputs.device)

        inputs_trans = inputs_rot + t  # B*N*3

        t_1 = inputs.mean(dim=1, keepdim=True)    # B*1*3
        inputs_ctrd_1 = inputs - t_1

        t_2 = inputs_trans.mean(dim=1, keepdim=True)    # B*1*3
        inputs_ctrd_2 = inputs_trans - t_2

        pts_d = {}
        pts_d['inputs_1'] = inputs
        pts_d['inputs_2'] = inputs_2
        pts_d['inputs_rot_2'] = inputs_rot
        pts_d['inputs_trans_2'] = inputs_trans
        pts_d['t_1'] = t_1
        pts_d['t_2'] = t_2
        pts_d['t'] = t
        pts_d['inputs_ctrd_1'] = inputs_ctrd_1
        pts_d['inputs_ctrd_2'] = inputs_ctrd_2

        # for key in pts_d:
        #     print(key, pts_d[key].shape)

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                if self.centralize:
                    inputs = self.preprocessor(inputs_ctrd_1)
                    inputs_rot = self.preprocessor(inputs_ctrd_2)
                else:
                    inputs = self.preprocessor(inputs)
                    inputs_rot = self.preprocessor(inputs_rot)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            if self.centralize:
                c = self.model.encode_inputs(inputs_ctrd_1)
                c_rot = self.model.encode_inputs(inputs_ctrd_2)
            else:
                c = self.model.encode_inputs(inputs)
                c_rot = self.model.encode_inputs(inputs_rot)

        # return c, c_rot, rot_d, inputs, inputs_rot, inputs_2
        return c, c_rot, rot_d, pts_d

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, z, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, z, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, z, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, z, c, **kwargs).logits

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, z, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, z, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, z, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, z, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        z, c = z.unsqueeze(0), c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, z, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, z, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), z, c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
