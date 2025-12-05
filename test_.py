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

import torch
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

# def save_pred_image(pred_mask, save_path):
#     """
#     pred_mask: [1, 2, H, W] tensor
#     save_path: Path object or string
#     """
#     # softmax 후 argmax -> [H, W] (0 or 1)
#     pred = torch.softmax(pred_mask, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()

#     # 예: 0 = 배경, 1 = 도로
#     color_map = {
#         0: [0, 0, 0],       # 검정 (배경)
#         1: [255, 0, 85],    # 자홍색 (도로)
#     }

#     # RGB 이미지로 변환
#     color_image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
#     for cls, color in color_map.items():
#         color_image[pred == cls] = color

#     save_path = Path(save_path)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(save_path), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
#     print(f"Saved prediction: {save_path}")

def test_orfd(
    dataset_root,
    sam_model,
    seg_decoder,
    num_workers=4,
    save_path = 'output_vits'
):
    # Dataset / Loader
    #train_ds = ORFDDataset(dataset_root, mode='training')
    test_ds = ORFDDataset(dataset_root, mode='testing')
    val_ds   = ORFDDataset(dataset_root, mode='validation')
    
    batch_size=1
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    dataset_loader = {#'train':train_loader,
                    #   'test':test_loader,
                      'validation':val_loader
                    }

    # 모델(이미 존재한다고 가정)
    # efficientsam, seg_decoder, preprocess, transform 은 사용자 코드에서 그대로 사용    
    transform = ResizeLongestSide(1024)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam_model.to(device).eval()
    seg_decoder.to(device).eval()

    # ====== Validation ======
    seg_decoder.eval()
    val_loss = 0.0
    val_iou_list = []
    running_loss = 0
    with torch.no_grad():
        for imgs, gts, base_name in dataset_loader['validation']:

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

            image_embedding = sam_model.image_encoder(input_image_torch)

            # decoder는 학습 대상 (grad 계산 O)
            pred_mask = seg_decoder(image_embedding) # [1, 2, 256, 256] 반환
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

            if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 크기와 gt_image 크기 비교
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            save_path = Path(save_path) 
            save_pred_image(pred_mask, save_path/base_name[0])

            # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
            gt = torch.as_tensor(gt, dtype=torch.long, device=pred_mask.device)
            if gt.ndim == 2:
                gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤

            # 🔥 0~255 → 0~1로 변환
            if gt.max() > 1:
                gt = (gt > 127).long()
            
            loss = torch.nn.functional.cross_entropy(pred_mask, gt)
            running_loss += loss.item() * gt.size(0)

            # IoU 계산
            preds = torch.softmax(pred_mask, dim=1).argmax(dim=1)  # [B,H,W] 0/1
            val_iou_list.append(binary_iou(preds, gt))

    val_loss = running_loss  / len(val_loader.dataset)
    mean_iou = np.mean(val_iou_list) if val_iou_list else 0.0

    print(f'Result:{val_loss}, Mean IOU:{mean_iou}')


def main():
    print("Loading model...")

    image_file = './ORFD_dataset'
    best_model_path = r'ckpts/best_vits_251118.pth'
    '''
    pth structure:
    {"epoch": epoch,
     "efficientsam": sam_model.state_dict(),
     "seg_decoder": seg_decoder.state_dict(),
     "optimizer": optimizer.state_dict(),
     "best_val_iou": best_val_iou}
    '''

    efficientsam = build_efficient_sam_vits()

    model_type = 'vits'
    seg_decoder = SegHead(sam_variant=model_type)

    checkpoint = torch.load(best_model_path)
    print(f"The best weight was found at {checkpoint['epoch']}")
    seg_decoder.load_state_dict(checkpoint['seg_decoder'])
   # 이미지 데이터 디렉터리 경로

    test_orfd(
        dataset_root = image_file,
        sam_model=efficientsam,
        seg_decoder=seg_decoder,
    )

if __name__ == '__main__':
    main()