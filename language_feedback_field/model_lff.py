import json
from typing import Dict, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType as TorchTensor

from nerfstudio.cameras.cameras import RayBundle
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.cameras.camera_paths import get_path_from_json

import sys
sys.path.append('/data/vision/torralba/scratch/gtangg12/conceptfields')
from conceptfields.renderer.pipeline import Pipeline, load_pipeline


DEFAULT_PROMPT = "Combine the following descriptions of surrounding objects into an informative description."


def generate_equiangular_rays(n: int, flatten=False, device=None) -> TorchTensor["n", "n", 3]:
    """
    :param n: angular resolution
    """
    theta, phi = torch.linspace(0, 2 * np.pi, n), torch.linspace(0, np.pi, n)
    theta_grid, phi_grid = torch.meshgrid(theta, phi)
    x = torch.sin(phi_grid) * torch.cos(theta_grid)
    y = torch.sin(phi_grid) * torch.sin(theta_grid)
    z = torch.cos(phi_grid)
    directions = torch.stack((x, y, z), axis=-1)
    if device is not None:
        directions = directions.to(device)
    return directions if not flatten else directions.reshape(-1, 3)


class ModelLFF(nn.Module):
    """
    Model for generating a description of a scene location.
    """
    def __init__(self, pipeline: Pipeline):
        """
        param checkpoint: Path to instance nerf checkpoint.
        """
        super().__init__()
        self.pipeline = pipeline
        

    def forward(self, pose: TorchTensor[4, 4], prompt=DEFAULT_PROMPT) -> Dict:
        """
        :param x: Tensor of shape [batch, 3] containing xyz coordinates of scene locations.
        :param prompt: Generate description from surrounding object descriptions.
        """
        # generate ray_bundle from xyz coords
        ray_bundle = self.sample_bundle(pose)
        outputs = self.pipeline.model(ray_bundle)
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        from collections import Counter
        freq = Counter()
        for i in outputs['semantics_pred']:
            freq[i.item()] += 1
        print(freq)
        mean_depth = Counter()
        for i, d in zip(outputs['semantics_pred'], outputs['depth']):
            mean_depth[i.item()] += d.item() / freq[i.item()]
        mean_depth = sorted(mean_depth.items(), key=lambda x: x[1])
        print(mean_depth)
        # get semantics and depth from querying model using ray_bundle: get vector of [n, 2] (semantics, depth)
        # filter by depth and unique(semantics[filtered indices])
        # ltm semantics based on prompt
        # return dict {description: str, objects: {object_id: depth}}
        return {}
    
    @classmethod
    def sample_bundle(cls, pose: TorchTensor[4, 4], n=32) -> RayBundle:
        """
        Sample spherical ray bundle from `poses` at resolution `n`.
        """
        directions = generate_equiangular_rays(n, flatten=True, device=pose.device)
        bundle = RayBundle(
            origins=pose[:3, 3:].reshape(1, 3).expand(len(directions), 3),
            directions=directions,
            pixel_area=torch.ones(len(directions), 1, device=pose.device),
        )
        bundle.set_camera_indices(0) # Needed for nerfacto appearance embedding
        return bundle
    

if __name__ == '__main__':
    BASEDIR = Path('/data/vision/torralba/scratch/gtangg12/conceptfields')
    cameras = json.load(open(BASEDIR / 'renders/camera_path_alignment.json', 'r'))
    cameras = get_path_from_json(cameras)

    pipeline = load_pipeline(
        config     =BASEDIR / 'outputs/semantics/nerfacto/2024-01-12_011604/config.yml',
        checkpoints=BASEDIR / 'outputs/semantics/nerfacto/2024-01-12_011604/nerfstudio_models',
    )
    model = ModelLFF(pipeline)
    model.cuda()
    for i in range(0, len(cameras.camera_to_worlds), len(cameras.camera_to_worlds) // 5):
        print(i)
        model(cameras.camera_to_worlds[i].cuda())