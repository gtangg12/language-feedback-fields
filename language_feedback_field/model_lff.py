import json
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from pathlib import Path

import deeplake
import numpy as np
from protocols import SingleOutput
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


def spherical_to_cartesian(theta: TorchTensor, phi: TorchTensor):
    """ 
    Convert spherical coordinates to cartesian coordinates 
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return x, y, z


def cartesian_to_spherical(x: TorchTensor, y: TorchTensor, z: TorchTensor) -> Tuple[float, float]:
    """ 
    Convert cartesian coordinates to spherical coordinates 
    """
    r     = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(z / r)
    phi   = torch.arctan2(y, x)
    return theta, phi


def average_angles_spherical(thetas: TorchTensor, phis: TorchTensor) -> Tuple[float, float]:
    """ 
    Compute the average of angles in spherical coordinates 
    """
    # Convert all angles to cartesian coordinates
    x, y, z = spherical_to_cartesian(thetas, phis)

    # Compute the mean of the cartesian coordinates
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    mean_z = torch.mean(z)
    
    # Convert the mean cartesian coordinates back to spherical coordinates
    return cartesian_to_spherical(mean_x, mean_y, mean_z)


def rad2deg(rad: float) -> float:
    """
    Convert radians to degrees.
    """
    return rad * 180 / np.pi


class ModelLFFOutputFunc:
    """
    """
    pass


class ModelLFFOutputFuncSceneContext(ModelLFFOutputFunc):
    """
    """
    def __init__(self, threshold_freq: int, threshold_distance: float):
        """
        """
        self.threshold_freq     = threshold_freq
        self.threshold_distance = threshold_distance

    def __call__(self, model_outputs: Dict, labels2descriptions: Dict) -> Dict[str, SingleOutput]:
        """
        """
        freq          = Counter()
        mean_distance = Counter()
        for i in model_outputs['semantics_pred']:
            freq[i.item()] += 1
        for i, distance in zip(model_outputs['semantics_pred'], model_outputs['depth']):
            mean_distance[i.item()] += distance.item() 
        for k, _ in mean_distance.items():
            mean_distance[k] /= freq[k]
        
        thetas, phis = model_outputs['angle']
        outputs = {}
        for label in freq.keys():
            if freq[label] < self.threshold_freq or mean_distance[label] > self.threshold_distance:
                continue
            theta, phi = average_angles_spherical(
                thetas[model_outputs['semantics_pred'].cpu() == label], 
                phis  [model_outputs['semantics_pred'].cpu() == label]
            )
            outputs[label] = {
                'description': labels2descriptions.get(label, 'No data available.'),
                'scoord': (mean_distance[label], rad2deg(theta.item()), rad2deg(phi.item())),
            }
        return outputs


class ModelLFF(nn.Module):
    """
    Model for generating a description of a scene location.
    """
    def __init__(self, ds: deeplake.Dataset, pipeline: Pipeline, output_func: ModelLFFOutputFunc):
        """
        param checkpoint: Path to instance nerf checkpoint.
        """
        super().__init__()
        self.pipeline = pipeline
        self.output_func = output_func
        self.labels2descriptions = ds.info.get('labels2descriptions', {})
        
    def forward(self, pose: TorchTensor[4, 4]) -> Dict[str, SingleOutput]:
        """
        :param pose: Camera pose.
        """
        # generate ray_bundle from xyz coords
        ray_bundle, theta, phi = self.sample_bundle(pose)
        model_outputs = self.pipeline.model(ray_bundle)
        model_outputs['angle'] = (theta, phi)
        return self.output_func(model_outputs, self.labels2descriptions)

    @classmethod
    def sample_bundle(cls, pose: TorchTensor[4, 4], n=32) -> RayBundle:
        """
        Sample spherical ray bundle from `poses` at resolution `n`.
        """
        theta, phi = torch.linspace(0, 2 * np.pi, n), torch.linspace(0, np.pi, n)
        theta, phi = torch.meshgrid(theta, phi)
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        directions = torch.stack((x, y, z), axis=-1).to(pose.device).reshape(-1, 3)
        theta, phi = theta.reshape(-1), phi.reshape(-1)

        bundle = RayBundle(
            origins=pose[:3, 3:].reshape(1, 3).expand(len(directions), 3),
            directions=directions,
            pixel_area=torch.ones(len(directions), 1, device=pose.device),
        )
        bundle.set_camera_indices(0) # Needed for nerfacto appearance embedding
        return bundle, theta, phi
    

if __name__ == '__main__':
    torch.cuda.set_device(4)
    from conceptfields.data.deeplake_utils import load_dataset
    ds = load_dataset('replica-niceslam/room0_grounded_sam')

    BASEDIR = Path('/data/vision/torralba/scratch/gtangg12/conceptfields')
    cameras = json.load(open(BASEDIR / 'renders/camera_path_navigation.json', 'r'))
    cameras = get_path_from_json(cameras)

    pipeline = load_pipeline(
        config     =BASEDIR / 'outputs/semantics/nerfacto/2024-01-12_011604/config.yml',
        checkpoints=BASEDIR / 'outputs/semantics/nerfacto/2024-01-12_011604/nerfstudio_models',
    )
    model = ModelLFF(ds, pipeline, output_func=ModelLFFOutputFuncSceneContext(10, 1.75))
    model.cuda()
    for i in range(0, len(cameras.camera_to_worlds), len(cameras.camera_to_worlds) // 5):
        print(i)
        model(cameras.camera_to_worlds[i].cuda())
    model(cameras.camera_to_worlds[-1].cuda())