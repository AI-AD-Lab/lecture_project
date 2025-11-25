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

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from torchvision import transforms
    
from dataloader import ORFDDataset
from torch.utils.data import DataLoader
from pathlib import Path

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        # if ann['pred_class'] == 1:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = [0.25526778, 0.19120787, 0.67079563, 0.35]
        img[m] = color_mask
    ax.imshow(img)


def show_anns_2(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        if ann['pred_class'] >= 0.9:
            m = ann['segmentation']
            img[m] = [0.25526778, 0.19120787, 0.67079563, 0.35]

    ax.imshow(img)

def build_optimizer(model, lr=1e-3):
    return AdamW(model.parameters(), lr=lr)

# 3) Poly LR scheduler: lr = base_lr * (1 - iter/max_iter) ** power
def build_poly_scheduler(optimizer, total_steps, power=0.9):
    def poly_decay(step):
        step = min(step, total_steps)
        return (1 - step / float(total_steps)) ** power
    return LambdaLR(optimizer, lr_lambda=poly_decay)

def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
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

'''
number of channels for each model
vitt = 192
vits = 384
vit_h = 1280
vit_l = 1024
vit_b = 768
'''

def compute_loss(logits, target):
    """
    logits: [B,2,H,W]  (배경/전경)
    target: [B,H,W]    (0/1, Long)
    """
    return F.cross_entropy(logits, target)

@torch.no_grad()
def binary_iou(pred_mask, target):
    """
    pred_mask: [B,2,H,W] (0/1)
    target   : [B,1,H,W] (0/1)
    """
    inter = (pred_mask & target).sum(dim=(1,2)).float()
    union = (pred_mask | target).sum(dim=(1,2)).float().clamp_min(1.0)
    return (inter / union).mean().item()

def extract_images_into_encoded_data(
    dataset_root,
    sam_model,
    device,
    model_type_name='vits'
):
    # Dataset / Loader

    phases = ['training', 'testing', 'validation']
    dataset = { phase:ORFDDataset(dataset_root, mode=phase) for phase in phases}
    dataloader = { phase:DataLoader(dataset[phase], batch_size=1, shuffle=False,
                              num_workers=1, drop_last=False) for phase in phases}

    sam_model.to(device).eval()
    transform = ResizeLongestSide(1024)

    save_path_dir = {phase:Path(f'./ORFD_dataset/{phase}/{model_type_name}') for phase in phases}
    
    for phase in dataloader.keys():
        
        encoding_data_save_dir = save_path_dir[phase]
        
        if not os.path.exists(encoding_data_save_dir):
            os.mkdir(encoding_data_save_dir)

        for imgs, gts, image_name in dataloader[phase]:
            # DataLoader가 batch_size=1일 때 rgb_image[0] 꺼내기
            rgb_image = imgs[0].numpy() if isinstance(imgs, torch.Tensor) else imgs
            gt = gts[0].numpy() if isinstance(gts, torch.Tensor) else gts

            # vits 기준 -> gb_image.shape => (720, 1280, 3), gt.shape=> (720, 1280) (세로, 가로, 채널)

            # 원본 크기
            ori_size = rgb_image.shape[:2] # (720, 1280)
            # transform 적용
            input_image = transform.apply_image(rgb_image)
            input_size = input_image.shape[:2] # (720,1280) -> (576,1024)로 변환 ResizeLongestSide(1024)

            # torch 텐서 변환 [1, 3, 576, 1024] -> [1, 3, 1024, 1024]
            input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()[None, :, :, :]
            input_image_torch = preprocess(input_image_torch).to(device)

            with torch.inference_mode(), torch.cuda.amp.autocast():
                image_embedding = sam_model.image_encoder(input_image_torch)

            if isinstance(image_embedding, tuple):
                image_embedding = image_embedding[0]

            emb_cpu = image_embedding.detach().cpu()


            img = image_name[0]
            dst_encoding_data_path = encoding_data_save_dir / img.replace('.png', '.pth')

            torch.save(emb_cpu, dst_encoding_data_path)

            del image_embedding, emb_cpu, input_image_torch
            torch.cuda.empty_cache()
            gc.collect()
            break



   

def main():

    model_types = [
        'vit_h',
        'vit_l', 
        'vit_b',
        'vits', 
        'vitt'
    ]
    device = torch.device("cuda")
    # device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

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

   # 이미지 데이터 디렉터리 경로
    image_file = './ORFD_dataset'
    for model_type in model_types:

        print(f'Now Loaddig.... {model_type}') # 모델 타입에 따른 설정
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])

        extract_images_into_encoded_data(
            dataset_root = image_file,
            sam_model=sam_model,
            device = device,
            model_type_name = model_type
        )

        break
if __name__ == '__main__':
    main()