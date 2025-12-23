import cv2  # type: ignore
from segment_anything import sam_model_registry
from tqdm import tqdm

import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from seg_decoder import SegHead
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F

import gc
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from torchvision import transforms
from adaptive_encoding_patch_model import RODSegAdaptivePatch
from dataloader import ORFDDataset
from torch.utils.data import DataLoader


def build_poly_scheduler(optimizer, total_steps, power=0.9):
    def poly_decay(step):
        step = min(step, total_steps)
        return (1 - step / float(total_steps)) ** power
    return LambdaLR(optimizer, lr_lambda=poly_decay)


def preprocess(x):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    x = x.to(torch.float32).to(pixel_mean.device)

    if x.shape[2] != 1024 or x.shape[3] != 1024:
        x = F.interpolate(
            x,
            (1024, 1024),
            mode="bilinear",
        )

    x = (x - pixel_mean) / pixel_std
    return x


def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        (64, 64),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


@torch.no_grad()
def binary_iou(pred_mask, target):
    """
    pred_mask: [B,H,W] (0/1)
    target   : [B,1,H,W] (0/1)
    """
    # pred_mask: [B,H,W], target: [B,1,H,W]
    pred_mask = pred_mask.bool()
    target = target.bool()

    inter = (pred_mask & target.squeeze(1)).sum(dim=(1, 2)).float()
    union = (pred_mask | target.squeeze(1)).sum(dim=(1, 2)).float().clamp_min(1.0)
    return (inter / union).mean().item()


def make_dataloaders(dataset_root, batch_size=1, num_workers=1):
    phases = ["training", "testing", "validation"]
    datasets = {phase: ORFDDataset(dataset_root, mode=phase) for phase in phases}
    loaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=batch_size,
            shuffle=(phase == "training"),
            num_workers=num_workers,
            drop_last=(phase == "training"),
        )
        for phase in phases
    }
    return loaders


def gt_processing(gt, device):
    # 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
    gt = torch.as_tensor(gt, dtype=torch.long, device=device)
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤

    #  0~255 → 0, 1로 변환
    if gt.max() > 1:
        gt = (gt > 128).long()
    return gt