import cv2  # type: ignore
from segment_anything import sam_model_registry
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

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

from tqdm import tqdm

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from torchvision import transforms
from adaptive_encoding_patch_model import RODSegAdaptivePatch
from dataloader import ORFDDataset
from torch.utils.data import DataLoader
from train_utils import (
    make_dataloaders, 
    binary_iou, 
    postprocess_masks, 
    preprocess, 
    build_poly_scheduler,
    gt_processing
)

import shutil

'''
인코딩 후에 학습하고 싶었으나, 인코딩 된 데이터 다 하지 않은 상태에서도
'''

def extract_latent_features(
    dataset_root,
    sam_model,
    device,
    model_type_name='vits',
    parent_dir = 'sam_latent_data',
):
    phases = ['training', 'validation', 'testing']

    # Dataset / Loader
    dataloaders = make_dataloaders(
        dataset_root=dataset_root,
        batch_size=1,
        num_workers=1
    )

    # 고정 SAM backbone
    sam_model.to(device).eval()

    transform = ResizeLongestSide(1024)

    parent_dir_model_type = Path(parent_dir) / model_type_name
    os.makedirs(parent_dir_model_type, exist_ok=True)

    # 각 phase 별로 공통 루프 사용
    for phase in phases:
        if phase not in dataloaders:
            continue

        phase_parent_dir = parent_dir_model_type / phase
        latent_feature_dir = phase_parent_dir / 'latent'

        src_gt_path = Path(dataset_root) / phase / 'gt_image'
        dst_gt_path = Path(phase_parent_dir) / 'gt_image'
        # gt_dir = phase_parent_dir / 'gt_image'

        dataloader = dataloaders[phase]

        if not os.path.exists(latent_feature_dir):
            os.makedirs(latent_feature_dir, exist_ok=True)
        
        if not os.path.exists(dst_gt_path):
            os.makedirs(dst_gt_path, exist_ok=True)

        for imgs, gts, _ in tqdm(dataloader, desc=phase):
            # 이미지 하나씩 처리 (ResizeLongestSide 때문에)
            rgb_image = imgs[0].numpy() if isinstance(imgs, torch.Tensor) else imgs
            gt = gts[0].numpy() if isinstance(gts, torch.Tensor) else gts

            # 원본 크기
            ori_size = rgb_image.shape[:2]  # (H, W)

            # transform 적용
            input_image = transform.apply_image(rgb_image)
            input_size = input_image.shape[:2]

            # [1, 3, H, W] 텐서로 변환 후 preprocess
            input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()[None, :, :, :]
            input_image_torch = preprocess(input_image_torch).to(device)

            # SAM image encoder는 항상 no_grad
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image_torch)


            latent = (image_embedding[0].cpu(), image_embedding[1])

            latent_feature_data_path = latent_feature_dir / _[0].replace('.png' , '.pt')
            torch.save(latent, latent_feature_data_path)

            src_png_path = src_gt_path / _[0].replace('.png' , '_fillcolor.png')
            dst_png_path = dst_gt_path / _[0].replace('.png' , '_fillcolor.png')
            shutil.copy(src_png_path, dst_png_path)


    
def main():
    model_types = [
        # 'vit_h',
        'vit_l', 
        'vit_b',
        'vits', 
        'vitt'
    ]
    device = torch.device("cuda:0")

    sam_model_dict = {
        "vit_h": sam_model_registry["vit_h"],
        "vit_l": sam_model_registry["vit_l"],
        "vit_b": sam_model_registry["vit_b"],
        "vits": build_efficient_sam_vits,
        "vitt": build_efficient_sam_vitt,
    }

    sam_model_weight_chekpoint = {
        "vit_h": "./weights/sam_vit_h_4b8939.pth",
        "vit_l": "./weights/sam_vit_l_0b3195.pth",
        "vit_b": "./weights/sam_vit_b_01ec64.pth",
        "vits": "./weights/efficient_sam_vits.pt",
        "vitt": "./weights/efficient_sam_vitt.pt",
    }

    image_file = "./ORFD_dataset"
    # date = datetime.now().strftime("%y%m%d")
    date = '251210'

    for model_type in model_types:
        print(f"Now Loading.... {model_type}")
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])
        seg_decoder = SegHead(sam_variant=model_type)
        adaptive_encoder = RODSegAdaptivePatch(model_type=model_type)

        extract_latent_features(
            dataset_root=image_file,
            sam_model=sam_model,
            device=device,
            parent_dir = f'ORFD_latent_data',
            model_type_name=model_type
        )

if __name__ == "__main__":
    main()
