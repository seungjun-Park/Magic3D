import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
from nerf.utils import safe_normalize, custom_meshgrid, get_rays

DIR_COLORS = np.array([
    [255, 0, 0, 255],  # front
    [0, 255, 0, 255],  # side
    [0, 0, 255, 255],  # back
    [255, 255, 0, 255],  # side
    [255, 0, 255, 255],  # overhead
    [0, 255, 255, 255],  # bottom
], dtype=np.uint8)


def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False,
               angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size) - 0.5) * 2.0,
                torch.rand(size),
                (torch.rand(size) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(theta, phi, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


class NeRFTrainDataset(Dataset):
    def __init__(self,
                 bound=1,
                 dt_gamma=0,
                 near=0.01,
                 far=1000,
                 radius_range=[1.0, 1.5],
                 theta_range=[45, 1-5],
                 phi_range=[-180, 180],
                 fovy_range=[40, 80],
                 default_radius=1.2,
                 defualt_theta=90,
                 default_phi=0,
                 defualt_fovy=60,
                 progressive_view=False,
                 progressive_level=False,
                 angle_overhead=30,
                 angle_front=60,
                 t_range=[0.02, 0.98],
                 H=256,
                 W=256,
                 size=100,
                 jitter_pose=None,
                 uniform_sphere_rate=None):
        super().__init__()

        self.bound = bound
        self.dt_gamma = dt_gamma

        self.radius_range = radius_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.fovy_range = fovy_range
        self.default_radius = default_radius
        self.default_theta = defualt_theta
        self.default_phi = default_phi
        self.default_fovy = defualt_fovy

        self.progressive_view = progressive_view
        self.progressive_level = progressive_level

        self.angle_overhead = angle_overhead
        self.angle_front = angle_front
        self.t_range = t_range

        self.H = H
        self.W = W
        self.size = size

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = near
        self.far = far  # infinite

        self.jitter_pose = jitter_pose
        self.uniform_sphere_rate = uniform_sphere_rate

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        B = len(index)  # always 1

        # random pose on the fly
        poses, dirs, thetas, phis, radius = rand_poses(B, radius_range=self.radius_range,
                                                       theta_range=self.theta_range,
                                                       phi_range=self.phi_range, return_dirs=True,
                                                       angle_overhead=self.angle_overhead,
                                                       angle_front=self.angle_front,
                                                       jitter=self.jitter_pose,
                                                       uniform_sphere_rate=self.uniform_sphere_rate)

        # random focal
        fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2 * focal / self.W, 0, 0, 0],
            [0, -2 * focal / self.H, 0, 0],
            [0, 0, -(self.far + self.near) / (self.far - self.near),
             -(2 * self.far * self.near) / (self.far - self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0)

        mvp = projection @ torch.inverse(poses)  # [1, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.default_theta
        delta_azimuth = phis - self.default_phi
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - self.default_radius

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
        }

        return data

class NeRFValidationDataset(Dataset):
    def __init__(self,
                 bound=1,
                 dt_gamma=0,
                 near=0.01,
                 far=1000,
                 radius_range=[1.0, 1.5],
                 theta_range=[45, 1-5],
                 phi_range=[-180, 180],
                 fovy_range=[40, 80],
                 default_radius=1.2,
                 defualt_theta=90,
                 default_phi=0,
                 defualt_fovy=60,
                 progressive_view=False,
                 progressive_level=False,
                 angle_overhead=30,
                 angle_front=60,
                 t_range=[0.02, 0.98],
                 H=256,
                 W=256,
                 size=100):
        super().__init__()

        self.bound = bound
        self.dt_gamma = dt_gamma

        self.radius_range = radius_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.fovy_range = fovy_range
        self.default_radius = default_radius
        self.default_theta = defualt_theta
        self.default_phi = default_phi
        self.default_fovy = defualt_fovy

        self.progressive_view = progressive_view
        self.progressive_level = progressive_level

        self.angle_overhead = angle_overhead
        self.angle_front = angle_front
        self.t_range = t_range

        self.H = H
        self.W = W
        self.size = size

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = near
        self.far = far  # infinite

        # [debug] visualize poses
        # poses, dirs, _, _, _ = rand_poses(100, self.device, radius_range=self.opt.radius_range, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def __len__(self):
        return 1

    def __getitem__(self, index):

        B = len(index)  # always 1

        # circle pose
        thetas = torch.FloatTensor([self.default_theta])
        phis = torch.FloatTensor([(index[0] / self.size) * 360])
        radius = torch.FloatTensor([self.default_radius])
        poses, dirs = circle_poses(radius=radius, theta=thetas, phi=phis, return_dirs=True,
                                   angle_overhead=self.angle_overhead, angle_front=self.angle_front)

        # fixed focal
        fov = self.default_fovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2 * focal / self.W, 0, 0, 0],
            [0, -2 * focal / self.H, 0, 0],
            [0, 0, -(self.far + self.near) / (self.far - self.near),
             -(2 * self.far * self.near) / (self.far - self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0)

        mvp = projection @ torch.inverse(poses)  # [1, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.default_theta
        delta_azimuth = phis - self.default_phi
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - self.default_radius

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
        }

        return data