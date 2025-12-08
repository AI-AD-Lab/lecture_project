import cv2  # type: ignore

from segment_anything import sam_model_registry
from tqdm import tqdm

import numpy as np
import torch
import glob
import time
import os
import matplotlib
import matplotlib.pyplot as plt
# from typing import Any, Dict, List
from typing import Any, Dict, List, Tuple

from seg_decoder import SegHead, SegHeadUpConv
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F

import torch, gc
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from torchvision import transforms
    
from dataloader import ORFDDataset
from torch.utils.data import DataLoader

# adpative vit backbone import
from effsam_vits_backbone_ap import (
    EffSamViTSBackboneAP,
    load_efficient_sam_vits_weights_ap,
)


class BoundaryScoreModule(nn.Module):
    """
    입력 RGB에서 Sobel gradient 기반으로 boundary score 계산.
    - 8x8 avg pool -> 2x2 max pool => 최종 16x16 patch grid와 동일한 해상도.
    - per-image 정규화 후 threshold로 boundary mask 생성.
    """

    def __init__(self, sub_patch: int = 8, thresh: float = 0.0):
        super().__init__()
        self.sub_patch = sub_patch
        self.thresh = thresh

        # Sobel 필터 (fixed kernel)
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32,
        )

        self.register_buffer(
            "kx",
            sobel_x.view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "ky",
            sobel_y.view(1, 1, 3, 3),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """
        x: [B,3,H,W] (입력 RGB)
        return: boundary mask [B,1,H/patch_size,W/patch_size]
        """
        B, C, H, W = x.shape

        # 1) grayscale
        if C == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = x[:, :1]  # 이미 single-channel이면

        # 2) Sobel gradient magnitude
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)  # [B,1,H,W]

        # 3) 8x8 sub-patch 단위로 평균(avg_pool)
        sp = self.sub_patch
        score_8 = F.avg_pool2d(
            grad_mag,
            kernel_size=sp,
            stride=sp,
        )  # [B,1,H/8,W/8]

        # 4) 2x2 max-pool로 16x16 patch grid와 맞추기
        score_16 = F.max_pool2d(score_8, kernel_size=2, stride=2)  # [B,1,H/16,W/16]

        # 5) per-image 정규화 후 thresholding
        B, _, h, w = score_16.shape
        flat = score_16.view(B, -1)
        mean = flat.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
        std = flat.std(dim=1, keepdim=True).view(B, 1, 1, 1)
        score_norm = (score_16 - mean) / (std + 1e-6)

        boundary_mask = (score_norm > self.thresh).float()  # [B,1,h,w]
        return boundary_mask, score_16, score_8, grad_mag


class RODSegAdaptivePatch(nn.Module):
    """
    EfficientSAM ViT-S backbone + Adaptive Patch boundary refinement decoder.
    인터페이스는 기존 RODSeg와 유사:
    - RODSegAdaptivePatch(sam_ckpt, num_classes, im_size, boundary_thresh)
    """

    def __init__(
        self,
        model_type = 'vits',
        boundary_thresh: float = 0.0,
    ):
        super().__init__()

        embed_size = {
            'vitt': 256, 'vits': 256,
            'vit_b': 256, 'vit_l': 256, 'vit_h': 256
        }

        embed_dim = embed_size[model_type]
        print(embed_dim)

        # 2) Boundary score 모듈
        self.boundary_module = BoundaryScoreModule(
            sub_patch=8,
            thresh=boundary_thresh,
        )

        # 3) local refine block (boundary 패치만 강화)
        self.local_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.patch_size = 16

    def forward(self, image, feature:torch.Tensor) -> torch.Tensor:
        x, out = feature          # x: [B,C,h,w]
        B, C, h, w = x.shape

        # 1) boundary mask는 RGB 기준으로 계산해야 함
        boundary_mask, _16, _8, _grad_mag = self.boundary_module(image, patch_size=self.patch_size)
        # boundary_mask: [B,1,h_b,w_b] (예: [B,1,h,w] 또는 더 coarse)

        # 2) feature 크기에 맞게 resize
        if boundary_mask.shape[-2:] != (h, w):
            boundary_mask = F.interpolate(
                boundary_mask,
                size=(h, w),
                mode="nearest",      # mask니까 nearest가 안전
            )

        # 3) 채널 방향 broadcast 준비
        if boundary_mask.shape[1] == 1:
            boundary_mask = boundary_mask.expand(-1, C, -1, -1)  # [B,1,h,w] -> [B,C,h,w]

        # 4) local refinement (feature 기반)
        local_res = self.local_refine(x)   # [B,C,h,w]
        # 5) boundary 위치만 강화
        feats_refined = x + local_res * boundary_mask  # [B,C,h,w]

        return feats_refined, out

    def boundary_analysis(self, image, feature:torch.Tensor) -> torch.Tensor:
        x, out = feature          # x: [B,C,h,w]
        B, C, h, w = x.shape

        # 1) boundary mask는 RGB 기준으로 계산해야 함
        boundary_mask, _16, _8, _grad_mag = self.boundary_module(image, patch_size=self.patch_size)
        # boundary_mask: [B,1,h_b,w_b] (예: [B,1,h,w] 또는 더 coarse)


        return boundary_mask, _16, _8, _grad_mag