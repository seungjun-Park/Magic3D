import torch
from torch.utils.data import Dataset
import datasets.scene as scene
class DreamfieldDataset(Dataset):
    def __init__(self,
                 th_range,
                 phi_range,
                 rad_range,
                 focal_mult_range,
                 retrieve_widths,
                 render_width,
                 lq_video_width=300.,
                 hq_video_width=400.,):
        super().__init__()
        self.hwf_clip_r = scene.scale_intrinsics(retrieve_widths[0])
        self.hwf_base = scene.scale_intrinsics(render_width)
        self.hwf_video = scene.scale_intrinsics(lq_video_width)
        self.hwf_video_hq = scene.scale_intrinsics(hq_video_width)

        self.th_range = th_range
        self.phi_range = phi_range
        self.rad_range = rad_range
        self.focal_mult_range = focal_mult_range

    def __len__(self):
        return

    def __getitem__(self, index):
        pose, rad, focal_mult = self.sample_pose_focal()

        rays = self.camera_ray_batch_base(pose, focal_mult)
        rays_in = self.shard_rays(rays)

        return

    def shard_rays(self, rays):
        return scene.shard_rays(rays)

    def camera_ray_batch_base(self, p, focal_mult):
        return scene.camera_ray_batch(p, *self.hwf_base[:2], self.hwf_base[2] * focal_mult)

    def sample_pose_focal(self):
        return scene.sample_camera(self.th_range, self.phi_range, self.rad_range, self.focal_mult_range)