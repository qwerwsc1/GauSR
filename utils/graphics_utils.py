#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def depths_double_to_points(view, depthmap1, depthmap2):
    W, H = view.image_width, view.image_height
    grid_x, grid_y = torch.meshgrid(
        (torch.arange(W, device="cuda", dtype=torch.float32) - view.Cx) / view.Fx,
        (torch.arange(H, device="cuda", dtype=torch.float32) - view.Cy) / view.Fy,
        indexing="xy",
    )
    rays_d = torch.stack(
        [grid_x, grid_y, torch.ones_like(grid_x)],
        dim=0,
    ).view(3, -1)
    rays_d.requires_grad_(False)
    points1 = depthmap1.reshape(1, -1) * rays_d
    points2 = depthmap2.reshape(1, -1) * rays_d
    return points1.reshape(3, H, W), points2.reshape(3, H, W)


def depth_double_to_normal(view, depth1, depth2):
    points1, points2 = depths_double_to_points(view, depth1, depth2)
    points = torch.stack([points1, points2], dim=0)
    dy = points[..., 2:, 1:-1] - points[..., :-2, 1:-1]
    dx = points[..., 1:-1, 2:] - points[..., 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=1), dim=1)
    output = torch.zeros_like(points)
    output[..., 1:-1, 1:-1] = normal_map
    return output


def depth_to_normal(view, depth):
    W, H = view.image_width, view.image_height
    grid_x, grid_y = torch.meshgrid(
        (torch.arange(W, device="cuda", dtype=torch.float32) - view.Cx) / view.Fx,
        (torch.arange(H, device="cuda", dtype=torch.float32) - view.Cy) / view.Fy,
        indexing="xy",
    )
    rays_d = torch.stack(
        [grid_x, grid_y, torch.ones_like(grid_x)],
        dim=0,
    )
    rays_d.requires_grad_(False)
    points = depth * rays_d
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=0), dim=0)
    output = torch.zeros_like(points)
    output[:, 1:-1, 1:-1] = normal_map
    return output