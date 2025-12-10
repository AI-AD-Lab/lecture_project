#%%
import cv2  # type: ignore

from segment_anything import sam_model_registry

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

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt , build_efficient_sam_vits
from efficient_sam_adaptive_patch.efficient_sam import  build_efficient_sam



from torchvision import transforms
    
from dataloader import ORFDDataset
from torch.utils.data import DataLoader

sam_model_dict ={
    'vit_h':sam_model_registry['vit_h'],
    'vit_l':sam_model_registry['vit_l'],
    'vit_b':sam_model_registry['vit_b'],
    'vits':build_efficient_sam_vits,
    'vitt':build_efficient_sam_vitt
}

sam_model_weight_chekpoint = {
    'vit_h': './weights/sam_vit_h_4b8939.pth',
    'vit_l': './weights/sam_vit_l_0b3195.pth',
    'vit_b': './weights/sam_vit_b_01ec64.pth',
    'vits': './weights/efficient_sam_vits.pt',
    'vitt': './weights/efficient_sam_vitt.pt',
}

model_type = 'vits'

vits = build_efficient_sam_vits()

new_adaptive_model = build_efficient_sam(
    encoder_patch_embed_dim=384,
    encoder_num_heads=6,
)


# print(new_adaptive_model.prompt_encoder)

# print('-'*50)
# print(vits.prompt_encoder)

# def print_structure(model, indent=0):
#     for name, module in model.named_children():
#         print(" " * indent + f"- {name}: {module.__class__.__name__}")
#         if len(list(module.children())) > 0:
#             print_structure(module, indent + 2)

# print_structure(new_adaptive_model)
with torch.no_grad():

    # encoder 복사
    w_base = vits.image_encoder.patch_embed.proj.weight    # [384, 3, 16, 16]
    b_base = vits.image_encoder.patch_embed.proj.bias      # [384] or None

    # 2) 새로운 8x8 patch_embed_8의 weight shape 확인
    w_fine = new_adaptive_model.image_encoder.patch_embed_8.proj.weight  # [384, 3, 8, 8]
    kH, kW = w_fine.shape[-2], w_fine.shape[-1] # (8,8)

    # 3) 16x16 → 8x8 bicubic interpolation
    w_init = F.interpolate(
        w_base,             # [384,3,16,16]
        size=(kH, kW),      # (8,8)
        mode="bicubic",
        align_corners=False,
    )

    new_adaptive_model.image_encoder.patch_embed_16.load_state_dict(vits.image_encoder.patch_embed.state_dict())
    new_adaptive_model.image_encoder.patch_embed_8.proj.weight.copy_(w_init)
    new_adaptive_model.image_encoder.blocks.load_state_dict(vits.image_encoder.blocks.state_dict())
    new_adaptive_model.image_encoder.neck.load_state_dict(vits.image_encoder.neck.state_dict())


    new_adaptive_model.prompt_encoder.load_state_dict(
        vits.prompt_encoder.state_dict()
    )

    # 2. mask_decoder 복사
    new_adaptive_model.mask_decoder.load_state_dict(
        vits.mask_decoder.state_dict()
    )

    torch.save(    {
        "model": new_adaptive_model.state_dict(),   # <-- sam.load_state_dict() 에 맞는 key
        "info": "Adaptive Patch SAM Model (patch16 + patch8 hybrid)",
        "base_checkpoint": "efficient_sam_vits.pt",}
        ,"./weights/adaptive_efficient_sam.pth")