import functools
import numpy as np
import torch

_HWF_BLENDER = np.array([800., 800., 1111.1111])

def scale_intrinsics(new_width, hwf=_HWF_BLENDER):
  """Scale camera intrinsics (heigh, width focal) to a desired image width."""
  return hwf * new_width / hwf[1]


def trans_t(t):
  return np.array([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, t],
      [0, 0, 0, 1],
  ], dtype=np.float32)


def rot_phi(phi):
  return np.array([
      [1, 0, 0, 0],
      [0, np.cos(phi), -np.sin(phi), 0],
      [0, np.sin(phi), np.cos(phi), 0],
      [0, 0, 0, 1],
  ], dtype=np.float32)


def rot_theta(th):
  return np.array([
      [np.cos(th), 0, -np.sin(th), 0],
      [0, 1, 0, 0],
      [np.sin(th), 0, np.cos(th), 0],
      [0, 0, 0, 1],
  ], dtype=np.float32)


def uniform_in_interval(interval):
  return np.random.uniform(size=()) * (interval[1] - interval[0]) + interval[0]


def pose_spherical(theta, phi, radius, precision=torch.float32):
  c2w = trans_t(radius)
  c2w = np.matmul(rot_phi(phi / 180. * np.pi), c2w, precision=precision)
  c2w = np.matmul(rot_theta(theta / 180. * np.pi), c2w, precision=precision)
  c2w = np.matmul(
      np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), c2w, precision=precision)
  return c2w


def sample_camera(th_range, phi_range, rad_range, focal_mult_range):
  """Sample random camera extrinsics (pose) on a sphere.

  NOTE: This function samples latitude and longitude uniformly in th_range and
        phi_range, respectively, but will oversample points from high latitudes,
        such as near the poles. In experiments, Dream Fields simply use
        phi_range = [-30, -30] to fix the camera to 30 degree elevation, so
        oversampling isn't a problem.

  Args:
    th_range (pair of floats): Camera azimuth range.
    phi_range (pair of floats): Camera elevation range. Negative values are
      above equator.
    rad_range (pair of floats): Distance to center of scene.
    focal_mult_range (pair of floats): Factor to multipy focal range.

  Returns:
    pose (array): Camera to world transformation matrix.
    rad (float): Radius of camera from center of scene.
    focal_mult (float): Value to multiply focal length by.
  """
  th = uniform_in_interval(th_range)
  phi = uniform_in_interval(phi_range)
  rad = uniform_in_interval(rad_range)
  focal_mult = uniform_in_interval(focal_mult_range)
  pose = pose_spherical(th, phi, rad)
  return pose, rad, focal_mult


def generate_rays(pixel_coords, pix2cam, cam2world):
  """Generate camera rays from pixel coordinates and poses."""
  homog = np.ones_like(pixel_coords[Ellipsis, :1])
  pixel_dirs = np.concatenate([pixel_coords + .5, homog], axis=-1)[Ellipsis, None]
  cam_dirs = np.matmul(pix2cam, pixel_dirs)
  ray_dirs = np.matmul(cam2world[Ellipsis, :3, :3], cam_dirs)[Ellipsis, 0]
  ray_origins = np.broadcast_to(cam2world[Ellipsis, :3, 3], ray_dirs.shape)

  dpixel_dirs = np.concatenate([pixel_coords + .5 + np.array([1, 0]), homog],
                               axis=-1)[Ellipsis, None]
  ray_diffs = np.linalg.norm(
      (np.matmul(pix2cam, dpixel_dirs) - cam_dirs)[Ellipsis, 0], axis=-1,
      keepdims=True) / np.sqrt(12.)
  return ray_origins, ray_dirs, ray_diffs


def pix2cam_matrix(height, width, focal):
  """Inverse intrinsic matrix for a pinhole camera."""
  return np.array([
      [1. / focal, 0, -.5 * width / focal],
      [0, -1. / focal, .5 * height / focal],
      [0, 0, -1.],
  ])

def camera_ray_batch(cam2world, height, width, focal):
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  pix2cam = pix2cam_matrix(height, width, focal)
  height, width = height.astype(int), width.astype(int)
  pixel_coords = np.stack(
      np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
  return generate_rays(pixel_coords, pix2cam, cam2world)

def shard_rays(rays, batch_shape=1, multihost=True):
  ray_origins, ray_dirs, ray_diffs = rays
  # if multihost:
  #   batch_shape = (jax.process_count(), jax.local_device_count(), -1)
  # else:
  #   batch_shape = (jax.local_device_count(), -1)

  return (np.reshape(ray_origins, batch_shape + ray_origins.shape[-1:]),
          np.reshape(ray_dirs, batch_shape + ray_dirs.shape[-1:]),
          np.reshape(ray_diffs, batch_shape + ray_diffs.shape[-1:]))