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

from pathlib import Path

#from seg_decoder import SegHead, SegHeadUpConv
#from segment_anything.utils.transforms import ResizeLongestSide
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

# adaptive patch 모델 적용
from adaptive_encoding_patch_model import RODSegAdaptivePatch

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




def visualization_ap(
    dataset_root: str,
    model: nn.Module,
    device: torch.device,
    phase: str = "validation",
    model_tag: str = "rod_effs_ap_vits",
    save_root_dir="./output_ap_canny/"
):
    """
    Adaptive Patch 모델(RODSegAdaptivePatch)로 ORFD를 시각화.
    - dataset_root: ORFDDataset 루트
    - model: 학습된 RODSegAdaptivePatch
    - phase: 'training' / 'testing' / 'validation' (기본 validation)
    - model_tag: 저장 폴더 이름에 들어갈 태그
    """
    ds = ORFDDataset(dataset_root, mode=phase)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    model.to(device).eval()

    # 출력 디렉토리 설정
    save_root_dir = Path(save_root_dir)
    base_dir = save_root_dir / model_tag



    mask_dir = base_dir / "masking_img"
    overlay_dir = base_dir / "masked_img"

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"[{phase}] Loaded {len(ds)} image pairs.")
    print(f"Saving to: {base_dir}")

    with torch.no_grad():
        for imgs, gts, image_name in loader:
            # imgs: [1,H,W,3] 또는 [1,3,H,W] (dataset 구현에 따라 다름)
            img_name = image_name[0]

            # 1) RGB 이미지 numpy로 (시각화용)
            if isinstance(imgs, torch.Tensor):
                rgb = imgs[0].cpu().numpy()
            else:
                rgb = imgs[0]

            # rgb가 [H,W,3] 인지 확인, 아니라면 transpose
            if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
                # [C,H,W] -> [H,W,C]
                rgb = np.transpose(rgb, (1, 2, 0))

            # 2) 모델 입력용 텐서 준비
            imgs_t = imgs.to(device, dtype=torch.float32)
            if imgs_t.ndim == 4 and imgs_t.shape[1] not in (1, 3):
                # [B,H,W,3] -> [B,3,H,W]
                imgs_t = imgs_t.permute(0, 3, 1, 2).contiguous()

            # 3) forward
            logits = model(imgs_t)  # [1,2,H,W]

            # 4) 저장 경로
            dst_mask_path = mask_dir / img_name
            dst_overlay_path = overlay_dir / img_name

            print(img_name)

            # 5) 마스크/오버레이 저장
            # save_pred_image(logits, dst_mask_path)
            # save_overlay_image(rgb, logits, save_path=dst_overlay_path)

def save_gray_normalized(pred_mask, save_path, eps=1e-6):
    """
    pred_mask: Tensor shape [1,1,H,W] or [B,1,H,W] similar gradient map
    저장 전 min/max normalization 수행 → 사람이 보이는 grayscale 이미지로 저장
    """

    # convert to numpy
    pred = pred_mask.detach().cpu().numpy().squeeze()  # shape [H,W]

    # min-max normalization
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + eps)

    # 스케일 조정 (0~255 uint8)
    img = (pred_norm * 255).astype(np.uint8)

    # 이미지 저장
    cv2.imwrite(save_path, img)

def binary_f1(preds, targets, eps=1e-6):
    """Binary F1-score (Dice) 계산 함수"""
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum().float()
    fp = (preds * (1 - targets)).sum().float()
    fn = ((1 - preds) * targets).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return 2 * (precision * recall) / (precision + recall + eps)

def adaptive_patch_analysis(
    dataset_root,
    sam_model,
    adaptive_encoder,
    seg_decoder,
    device,
    model_type_name='vits',
    save_path_dir='ckpt_analysis'
):

    phases = [
        'testing' 
    ]

    # Dataset / Loader
    dataset = { phase:ORFDDataset(dataset_root, mode=phase) for phase in phases}
    dataset_loader = { phase:DataLoader(dataset[phase], batch_size=1, shuffle=True,
                              num_workers=1, drop_last=True) for phase in phases}

    # 모델(이미 존재한다고 가정)
    # efficientsam, seg_decoder, preprocess, transform 은 사용자 코드에서 그대로 사용
    sam_model.to(device).eval()
    adaptive_encoder.to(device).eval()
    seg_decoder.to(device).eval()

    transform = ResizeLongestSide(1024)

    print(f'Now Visualizaing... {model_type_name}')

    parent_dir = Path(save_path_dir) / model_type_name

    mask_image_save_dir = parent_dir / 'masking_img'
    masked_gt_image_dir = parent_dir / 'masked_img'
    boundary_image_mask_dir = parent_dir / 'boundary_img_mask'
    boundary_image_8 = parent_dir / 'boundary_img_8'
    boundary_image_16 = parent_dir / 'boundary_img_16'
    boundary_image_gray_mag = parent_dir / 'boundary_gray_mag'
    # boundary_image_masked_dir = parent_dir / 'boundary_img_masked'

    path_list = [
        mask_image_save_dir,
        masked_gt_image_dir,
        boundary_image_mask_dir,
        boundary_image_8,
        boundary_image_16,
        boundary_image_gray_mag,
        # boundary_image_masked_dir
    ]

    for path_dir in path_list:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
       
    test_running_loss = 0.0
    test_iou_list = []
    test_f1_list = []
    frame_times = []

    for imgs, gts, _ in tqdm(dataset_loader['testing']):

        frame_start  = time.time()

        # DataLoader가 batch_size=1일 때 rgb_image[0] 꺼내기
        rgb_image = imgs[0].numpy() if isinstance(imgs, torch.Tensor) else imgs
        gt = gts[0].numpy() if isinstance(gts, torch.Tensor) else gts

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


            ap_image_embedding = adaptive_encoder(input_image_torch, image_embedding)
            boundary, _8, _16, _gray= adaptive_encoder.boundary_analysis(input_image_torch, image_embedding)

            pred_mask = seg_decoder(ap_image_embedding) # [1, 2, 256, 256] 반환
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

            if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 크기와 gt_image 크기 비교
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )

            dst_masking_image_save_path = mask_image_save_dir / _[0]
            dst_overlay_image_save_path = masked_gt_image_dir / _[0]
            dst_boundary_image_mask_save_path = boundary_image_mask_dir / _[0]
            dst_boundary_image_8_save_path = boundary_image_8 / _[0]
            dst_boundary_image_16_save_path = boundary_image_16 / _[0]
            dst_boundary_image_gray_mag_save_path = boundary_image_gray_mag / _[0]
            # dst_boundary_image_masked_save_path = boundary_image_masked_dir / _[0]

            save_pred_image(pred_mask, dst_masking_image_save_path )
            save_overlay_image(rgb_image ,pred_mask, save_path = dst_overlay_image_save_path)

            save_gray_normalized(boundary , dst_boundary_image_mask_save_path)
            save_gray_normalized(_8 , dst_boundary_image_8_save_path)
            save_gray_normalized(_16 , dst_boundary_image_16_save_path)
            save_gray_normalized(_gray , dst_boundary_image_gray_mag_save_path)

            # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
            gt = torch.as_tensor(gt, dtype=torch.long, device=pred_mask.device)
            if gt.ndim == 2:
                gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤

            # 🔥 0~255 → 0~1로 변환
            if gt.max() > 1:
                gt = (gt > 127).long()

        frame_end = time.time()
        frame_times.append(frame_end - frame_start)

        # ---- Loss ----
        loss = torch.nn.functional.cross_entropy(pred_mask, gt)
        test_running_loss += loss.item() * gt.size(0)

        # ---- IoU 계산 ----
        preds = torch.softmax(pred_mask, dim=1).argmax(dim=1)
        test_iou_list.append(binary_iou(preds, gt))
        # ---- F1-score 계산 추가 ----
        f1 = binary_f1(preds, gt)
        test_f1_list.append(f1.item())

    end_time = time.time()
    mean_test_loss = test_running_loss / len(dataset_loader['testing'])
    mean_test_iou = sum(test_iou_list) / len(test_iou_list)
    mean_test_f1 = sum(test_f1_list) / len(test_f1_list)

    avg_time = sum(frame_times) / len(frame_times)
    fps = 1.0 / avg_time

    total_time = sum(frame_times)

    print(f'Testing Time: {total_time:.2f} seconds, FPS: {fps:.2f}')
    print(f'Test Loss: {mean_test_loss:.4f}, Test mIoU: {mean_test_iou:.4f}, Test F1-score: {mean_test_f1:.4f}')

    test_mean_loss = test_running_loss / len(dataset_loader['testing'].dataset)
    test_miou = np.mean(test_iou_list) if test_iou_list else 0.0
    print(f'TEST || Loss - {test_mean_loss}, mIoU - {test_miou}')

    del sam_model
    del seg_decoder
    del adaptive_encoder
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")





def main():
    model_types = [
        'vit_h',
        'vit_l', 
        'vit_b',
        'vits', 
        'vitt'
    ]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    seghead_adaptive_patch_model_weight_chekpoint = {
        'vit_h': './ckpts_adaptive_canny/vit_h_best_val.pth',
        'vit_l': './ckpts_adaptive_canny/vit_l_best_val.pth',
        'vit_b': './ckpts_adaptive_canny/vit_b_best_val.pth',
        'vits': './ckpts_adaptive_canny/vits_best_val.pth',
        'vitt': './ckpts_adaptive_canny/vitt_best_val.pth',  
    }




   # 이미지 데이터 디렉터리 경로
    dataset_root = './ORFD_dataset'

    for model_type in model_types:

        print(f'Now Loaddig.... {model_type}') # 모델 타입에 따른 설정

        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])
        
        checkpoint = torch.load(seghead_adaptive_patch_model_weight_chekpoint[model_type])

        seg_decoder = SegHead(sam_variant=model_type)
        seg_decoder.load_state_dict(checkpoint["seg_decoder"])
        
        adaptive_encoder = RODSegAdaptivePatch(model_type=model_type)
        adaptive_encoder.load_state_dict(checkpoint["adaptive_encoder"])


        print(f"The best weight was found at {checkpoint['epoch']}")



        adaptive_patch_analysis(
            dataset_root= dataset_root,
            sam_model=sam_model,
            adaptive_encoder = adaptive_encoder,
            seg_decoder = seg_decoder,
            device = device,
            model_type_name=model_type,
            save_path_dir='output_ap_canny'
        )



if __name__ == '__main__':
    main()