# 파일 위치:
# models/transformer_models/backbones/effsam_vits_backbone_ap.py


# import ViTEncoder, 
import torch.nn as nn
from effsam_vits_backbone import (
    ViTSEncoder,
    load_efficient_sam_vits_weights,
)

class EffSamViTSBackboneAP(nn.Module):
    """
    Adaptive Patch 버전에서 쓸 EfficientSAM ViT-S 백본 래퍼.
    내부 구조는 기존 EffSamViTSBackbone과 동일하게 ViTSEncoder만 사용.
    """

    def __init__(self, img_size: int = 1024):
        super().__init__()
        self.encoder = ViTSEncoder(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
        )

    def forward(self, x):
        # x: [B,3,H,W]
        return self.encoder(x)  # [B,384,H/16,W/16]


def load_efficient_sam_vits_weights_ap(model: EffSamViTSBackboneAP, ckpt_path: str):
    """
    기존 로더를 그대로 재사용.
    """
    return load_efficient_sam_vits_weights(model, ckpt_path)
