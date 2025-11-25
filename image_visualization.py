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

def save_pred_image(pred_mask, save_path):
    pred = torch.softmax(pred_mask, dim=1).argmax(dim=1)  # [1, 720, 1280]
    pred = pred[0].detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite(save_path, pred * 255)

def save_overlay_image(rgb_image, pred_mask, save_path, color=(0, 0, 255), alpha=0.5):
    """
    rgb_image: 원본 RGB 이미지 (H, W, 3)
    pred_mask: [1, 2, H, W] or [2, H, W] tensor
    save_path: 저장 경로 (str or Path)
    color: 덮어쓸 색 (BGR) - 기본 빨강
    alpha: 투명도 (0~1)
    """

    # 1) 소프트맥스 → argmax로 이진 마스크 생성
    if pred_mask.dim() == 4:
        pred = torch.softmax(pred_mask, dim=1).argmax(dim=1)[0]
    else:
        pred = torch.softmax(pred_mask.unsqueeze(0), dim=1).argmax(dim=1)[0]

    pred = pred.detach().cpu().numpy().astype(np.uint8)

    # 2) 원본 이미지는 BGR 기반으로 변환 (cv2는 BGR)
    overlay_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 3) 빈 컬러 마스크 생성
    mask_color = np.zeros_like(overlay_img, dtype=np.uint8)
    mask_color[pred == 1] = color  # 예측된 부분에 색칠

    # 4) 원본 이미지 위에 반투명 오버레이
    output = cv2.addWeighted(overlay_img, 1.0, mask_color, alpha, 0)

    # 5) 저장
    cv2.imwrite(str(save_path), output)

def visualization(
    dataset_root,
    sam_model,
    seg_decoder,
    device,
    model_type_name='vits'
):
    # Dataset / Loader

    phases = [
        # 'training', 
        # 'testing', 
        'validation'
    ]

    dataset = { phase:ORFDDataset(dataset_root, mode=phase) for phase in phases}
    dataloader = { phase:DataLoader(dataset[phase], batch_size=1, shuffle=False,
                              num_workers=1, drop_last=False) for phase in phases}

    sam_model.to(device).eval()
    seg_decoder.to(device).eval()

    transform = ResizeLongestSide(1024)
    best_val_iou = 0.0

    save_path_dir = {phase:Path(f'./output/{model_type_name}') for phase in phases}
    
    for phase in dataloader.keys():
        encoding_data_save_dir = save_path_dir[phase]
        mask_image_save_dir = encoding_data_save_dir / 'masking_img'
        masked_gt_image_dir = encoding_data_save_dir / 'masked_img'
        
        if not os.path.exists(mask_image_save_dir):
            os.makedirs(mask_image_save_dir, exist_ok=True)

        if not os.path.exists(masked_gt_image_dir):
            os.makedirs(masked_gt_image_dir, exist_ok=True)

        running_loss = 0
        for imgs, gts, image_name in dataloader[phase]:
            # DataLoader가 batch_size=1일 때 rgb_image[0] 꺼내기
            rgb_image = imgs[0].numpy() if isinstance(imgs, torch.Tensor) else imgs
            gt = gts[0].numpy() if isinstance(gts, torch.Tensor) else gts

            img = image_name[0]
            dst_masking_image_save_path = mask_image_save_dir / img
            dst_overlay_image_save_path = masked_gt_image_dir / img
            # 원본 크기
            ori_size = rgb_image.shape[:2] # (720, 1280)
            # transform 적용
            input_image = transform.apply_image(rgb_image)
            input_size = input_image.shape[:2] # (720,1280) -> (576,1024)로 변환 ResizeLongestSide(1024)

            # torch 텐서 변환 [1, 3, 576, 1024] -> [1, 3, 1024, 1024]
            input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()[None, :, :, :]
            input_image_torch = preprocess(input_image_torch).to(device)

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image_torch)
                pred_mask = seg_decoder(image_embedding) # [1, 2, 256, 256] 반환
                pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

            if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 크기와 gt_image 크기 비교
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )

            save_pred_image(pred_mask, dst_masking_image_save_path )
            save_overlay_image(rgb_image ,pred_mask, save_path = dst_overlay_image_save_path)

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

    seg_decoder_weight_checkpoint = {
        'vit_h': './ckpts/best_vit_h_1118.pth',
        'vit_l': './ckpts/best_vit_l_1118.pth',
        'vit_b': './ckpts/best_vit_b_1118.pth',
        'vits': './ckpts/best_vits_1118.pth',
        'vitt': './ckpts/best_vitt_1118.pth',  
    }

   # 이미지 데이터 디렉터리 경로
    image_file = './ORFD_dataset'
    for model_type in model_types:

        print(f'Now Loaddig.... {model_type}') # 모델 타입에 따른 설정
        
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])
        seg_decoder = SegHead(sam_variant=model_type)
        checkpoint = torch.load(seg_decoder_weight_checkpoint[model_type], weights_only=False)
        seg_decoder.load_state_dict(checkpoint['seg_decoder'])
        print(f"The best weight was found at {checkpoint['epoch']}")

        visualization(
            dataset_root = image_file,
            sam_model=sam_model,
            seg_decoder = seg_decoder,
            device = device,
            model_type_name = model_type
        )

if __name__ == '__main__':
    main()