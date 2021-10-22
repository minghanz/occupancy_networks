'''Make sure that axis_angle_to_matrix() and so3.exp() are equivalent'''
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations, axis_angle_to_matrix
import so3

import numpy as np
import torch

if __name__ == "__main__":
    axis_angle = np.random.randn(3)
    axis_angle = torch.from_numpy(axis_angle)
    rot_mat_1 = axis_angle_to_matrix(axis_angle)
    rot_mat_2 = so3.exp(axis_angle)
    print(rot_mat_1)
    print(rot_mat_2)
    if torch.equal(rot_mat_1, rot_mat_2):
        print("The results of the two functions are IDENTICAL. ")
    elif torch.allclose(rot_mat_1, rot_mat_2):
        print("The results of the two functions are CLOSE. ")   # this is the result
    else:
        print("The results of the two functions are DIFFERENT. ")