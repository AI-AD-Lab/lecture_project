# 파일 위치:
# models/rod_effs_ap_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return boundary_mask


class RODSegAdaptivePatch(nn.Module):
    """
    EfficientSAM ViT-S backbone + Adaptive Patch boundary refinement decoder.
    인터페이스는 기존 RODSeg와 유사:
    - RODSegAdaptivePatch(sam_ckpt, num_classes, im_size, boundary_thresh)
    """

    def __init__(
        self,
        sam_ckpt: str,
        num_classes: int = 2,
        im_size: int = 1024,
        boundary_thresh: float = 0.0,
    ):
        super().__init__()

        # 1) ViT-S backbone
        self.backbone = EffSamViTSBackboneAP(img_size=im_size)

        if sam_ckpt is not None and sam_ckpt != "":
            try:
                load_efficient_sam_vits_weights_ap(self.backbone, sam_ckpt)
            except Exception as e:
                print("[WARN] efficient_sam_vits 로드 실패 (Adaptive Patch), 랜덤 초기화 사용")
                print("      ", e)

        # encoder freeze (원하면 여기 주석 처리해서 backbone도 학습할 수 있음)
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = 384  # ViT-S

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

        # 4) decoder (기존과 유사)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
        )

        self.patch_size = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]
        return: [B,num_classes,H,W]
        """
        B, _, H, W = x.shape

        # 1) backbone feature
        feats = self.backbone(x)  # [B,384,h,w], h=H/16, w=W/16
        _, C, h, w = feats.shape

        # 2) boundary mask (RGB 기반)
        boundary_mask = self.boundary_module(x, patch_size=self.patch_size)  # [B,1,h,w]

        # 3) local refinement (boundary 위치만)
        local_res = self.local_refine(feats)          # [B,C,h,w]
        # [B,1,h,w] * [B,C,h,w] -> broadcasting으로 채널 방향 확장
        feats_refined = feats + local_res * boundary_mask

        # 4) decoder -> upsample
        logits = self.decoder(feats_refined)          # [B,num_classes,h,w]
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return logits
