import json
import os
import random
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Type, Optional

import torch
import torch.nn as nn
from torchvision.ops import masks_to_boxes
from torchvision.transforms import ToPILImage
from torchtyping import TensorType as TorchTensor
from tqdm import tqdm

from nerfstudio.configs.base_config import InstantiateConfig

import sys
sys.path.append('/data/vision/torralba/scratch/gtangg12/conceptfields')
from conceptfields.data.deeplake_utils import load_tensor_at
from conceptfields.data.processors.model_gpt import GPT, SystemMode
from conceptfields.data.processors.processor_sequence_base import BaseSequenceProcessor
from conceptfields.utils.segmentation import crop, expand_bbox


VLM_SYSTEM_TEXT = """You are given two images of an object in an indoor scene. The first is a crop of the object. The second is a view of the object zoomed out with scene context. 
You should describe the object and its relationship to its surroundings using ONE OR TWO CONCISE SENTENCES.

EXAMPLE:

First image shows a close-up of potted lush banana plant. Second image shows a view of the plant in the corner of a room next to a window and a floor lamp
Output: A lush banana plant in a corner of a room next to a window and a floor lamp."""


LLM_SYSTEM_TEXT = """You are given a sequence of possibly redundant captions. You should summarize the sequence of captions as ONE CONCISE PARAGRAPH. Filter out captions that don't appear to describe the object. 

Do not use flowerly language as you are speaking to a robot. Begin you response by denoting the object you are describing. 

EXAMPLE:

A modern living room setup with a round coffee table at the center, surrounded by a white sofa, round side tables, and a decorative floor lamp.
A round wooden coffee table in the center of a living room surrounded by white sofas and a patterned area rug.
A round wooden coffee table is situated between two white sofas in a well-furnished living room.
A wooden round coffee table situated between two white sofas in a living room.
A contemporary living room with a round wooden coffee table in the center, surrounded by a white couch, chairs, and accompanying side tables

Output: Object: A round wooden coffee table. The coffee table is in the center of a living room surrounded by white sofas and a patterned area rug.
"""


@dataclass
class SequenceCaptionConfig(InstantiateConfig):

    """ Target to instantiate """
    _target: Type = field(default_factory=lambda: SequenceCaption)

    """ Models for VLM, LLM, and CLIP """
    model_vlm : nn.Module = field(default_factory=lambda: GPT(system_mode=SystemMode.MAIN))
    model_llm : nn.Module = field(default_factory=lambda: GPT(system_mode=SystemMode.JSON))
    

class SequenceCaption(BaseSequenceProcessor):
    """
    Generate captions for objects in a scene given as sequence of images with 3D consistent instance labels.
    """
    def setup_module(self):
        """
        Setup VLM, LLM, and sample indices.
        """
        assert torch.cuda.is_available(), 'SequenceCaption requires CUDA.'
        self.model_vlm = self.config.model_vlm
        self.model_llm = self.config.model_llm

    def forward(self, load_from: Optional[Path]=None) -> Dict:
        """
        Returns dict mapping label to frame sequence fused caption.
        """
        if load_from and os.path.exists(load_from):
            label2texts = json.load(open(load_from, 'r'))
        else:
            label2texts = self.compute_label2texts()
            json.dump(label2texts, open(load_from, 'w'))

        for label, texts in tqdm(label2texts.items(), desc='Fusing label captions'):
            label2texts[label] = self.call_llm(texts)
        self.ds.info['label2texts'] = label2texts
        return label2texts
    
    def compute_label2texts(self) -> Dict:
        """
        """
        label2indices = defaultdict(list)
        indices2label = defaultdict(list)
        for i in range(len(self.ds)):
            imask = load_tensor_at(self.ds, 'sam/sequence_segmentation/mask', i)
            labels, counts = torch.unique(imask, return_counts=True)
            for l, c in zip(labels, counts):
                if l == 0: continue
                label2indices[l.item()].append((c.item(), i))
        for label, indices in label2indices.items():
            label2indices[label] = sorted(indices, key=lambda x: x[0], reverse=True)[:5]
            label2indices[label] = list(filter(lambda x: x[0] > 1024, label2indices[label]))
        for label, indices in label2indices.items():
            for _, i in indices:
                indices2label[i].append(label)

        label2texts = defaultdict(list)
        for i, labels in tqdm(indices2label.items(), desc='Generating captions'):
            imask  = load_tensor_at(self.ds, 'sam/sequence_segmentation/mask', i)
            image  = load_tensor_at(self.ds, 'image', i)
            bmasks = torch.stack([imask == l for l in labels])
            bboxes = masks_to_boxes(bmasks)
            crops_item  = [crop(image, expand_bbox(b, scale=1.0, image_dim=image.shape[:2])) for b in bboxes]
            crops_scene = [crop(image, expand_bbox(b, scale=1.5, image_dim=image.shape[:2])) for b in bboxes]
            for label, ci, cs in zip(labels, crops_item, crops_scene):
                label2texts[label].append(self.call_vlm([ci, cs]))
        print(f'Found {len(label2texts)} labels.')

        return label2texts

    def call_vlm(self, images: List[TorchTensor["H", "W", "C"]]) -> List[str]:
        """
        Given a multiple images, return caption that combines all images.
        """
        self.model_vlm.reset()
        return self.model_vlm(image=[ToPILImage()(x.permute(2, 0, 1)) for x in images])
    
    def call_llm(self, texts: List[str], max_count=5) -> str:
        """
        Helper to fuse list of captions.
        """
        if len(texts) > max_count:
            texts = random.sample(texts, max_count)
        self.model_llm.reset()
        return self.model_llm(text='\n'.join(texts))


if __name__ == '__main__':
    import deeplake
    ds = deeplake.load('/data/vision/torralba/scratch/gtangg12/data/conceptfields/replica-niceslam/room0_grounded_sam')
    model = SequenceCaptionConfig(
        model_vlm=GPT(
            system_mode=SystemMode.MAIN,
            system_text=VLM_SYSTEM_TEXT,
        ),
        model_llm=GPT(
            system_mode=SystemMode.MAIN,
            system_text=LLM_SYSTEM_TEXT,
        ),
    ).setup(ds=ds)
    model(load_from='/data/vision/torralba/scratch/gtangg12/language-feedback-fields/language_feedback_field/cache.json')