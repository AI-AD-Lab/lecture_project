import cv2  # type: ignore

from segment_anything import sam_model_registry
from tqdm import tqdm
import csv
from datetime import datetime

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
from adaptive_encoding_patch_model import RODSegAdaptivePatch
from dataloader import ORFDDataset
from torch.utils.data import DataLoader

from logger import TrainLogger

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



def train_orfd(
    dataset_root,
    sam_model,
    adaptive_encoder,
    seg_decoder,
    device,
    epochs=100,
    lr = 1e-4,
    model_type_name='vits',
    date_time='250000'
):

    phases = [
        'training', 
        'testing', 
        'validation'
    ]


    # Dataset / Loader
    dataset = { phase:ORFDDataset(dataset_root, mode=phase) for phase in phases}
    dataset_loader = { phase:DataLoader(dataset[phase], batch_size=1, shuffle=True,
                              num_workers=1, drop_last=True) for phase in phases}

    # 모델(이미 존재한다고 가정)
    # efficientsam, seg_decoder, preprocess, transform 은 사용자 코드에서 그대로 사용
    sam_model.to(device).eval()
    adaptive_encoder.to(device).train()
    seg_decoder.to(device).train()

    # Optim / Scheduler
    # params = list(efficientsam.parameters()) + list(seg_decoder.parameters())
    #seg_decoder만 학습
    # optimizer = torch.optim.AdamW(seg_decoder.parameters(), lr=lr, weight_decay=1e-3)

    optimizer = torch.optim.AdamW(
    [
        {"params": seg_decoder.parameters(), "lr": lr},          # decoder → full LR
        {"params": adaptive_encoder.parameters(), "lr": lr},  # encoder → 작은 LR
    ], weight_decay=1e-3)

    total_steps = len(dataset_loader['training']) * epochs
    scheduler = build_poly_scheduler(optimizer, total_steps, power=0.9)
    transform = ResizeLongestSide(1024)

    print(f'Train {model_type_name}')

    early_stop_count = 0
    best_test_miou = 0
    best_val_miou = 0

    parent_dir = 'ckpts_adaptive'
    log_path = f"ckpts_adaptive/{model_type_name}_train_log_{date_time}.txt"

    logger  = TrainLogger(log_path)

    for epoch in range(1, epochs+1):
        print(f'processing: {epoch}/{epochs+1}')

        train_running_loss = 0.0
        train_iou_list = []
        for imgs, gts, _ in tqdm(dataset_loader['training']):
            seg_decoder.train()
            adaptive_encoder.train()
            
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

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image_torch)

            ap_image_embedding = adaptive_encoder(input_image_torch, image_embedding)

            # decoder는 학습 대상 (grad 계산 O)
            pred_mask = seg_decoder(ap_image_embedding) # [1, 2, 256, 256] 반환
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

            if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 크기와 gt_image 크기 비교
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )

            # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
            gt = torch.as_tensor(gt, dtype=torch.long, device=pred_mask.device)
            if gt.ndim == 2:
                gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤

            # 🔥 0~255 → 0~1로 변환
            if gt.max() > 1:
                gt = (gt > 127).long()

            loss = torch.nn.functional.cross_entropy(pred_mask, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item() * gt.size(0)

            # IoU 계산
            preds = torch.softmax(pred_mask, dim=1).argmax(dim=1) 
            train_iou_list.append(binary_iou(preds, gt))

        train_mean_loss = train_running_loss / len(dataset_loader['training'].dataset)
        train_miou = np.mean(train_iou_list) if train_iou_list else 0.0
        print(f'TRAIN || Loss - {train_mean_loss}, mIoU - {train_miou}')

        val_running_loss = 0.0
        val_iou_list = []
        for imgs, gts, _ in tqdm(dataset_loader['validation']):
            seg_decoder.eval()
            adaptive_encoder.eval()

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

            # decoder는 학습 대상 (grad 계산 O)
            pred_mask = seg_decoder(ap_image_embedding) # [1, 2, 256, 256] 반환
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

            if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 크기와 gt_image 크기 비교
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )

            # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
            gt = torch.as_tensor(gt, dtype=torch.long, device=pred_mask.device)
            if gt.ndim == 2:
                gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤

            # 🔥 0~255 → 0~1로 변환
            if gt.max() > 1:
                gt = (gt > 127).long()

            loss = torch.nn.functional.cross_entropy(pred_mask, gt)
            val_running_loss += loss.item() * gt.size(0)

            # IoU 계산
            preds = torch.softmax(pred_mask, dim=1).argmax(dim=1) 
            val_iou_list.append(binary_iou(preds, gt))

        val_mean_loss = val_running_loss / len(dataset_loader['validation'].dataset)
        val_miou = np.mean(val_iou_list) if val_iou_list else 0.0
        print(f'VAL || Loss - {val_mean_loss}, mIoU - {val_miou}')


        if best_val_miou > val_miou:
            early_stop_count += 1
            if early_stop_count > 10:
                print('early stop activated')
                break
        else:
            best_val_miou = val_miou

        test_running_loss = 0.0
        test_iou_list = []
        for imgs, gts, _ in tqdm(dataset_loader['testing']):

            seg_decoder.eval()
            adaptive_encoder.eval()

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

            # decoder는 학습 대상 (grad 계산 O)
            pred_mask = seg_decoder(ap_image_embedding) # [1, 2, 256, 256] 반환
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

            if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 크기와 gt_image 크기 비교
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )

            # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
            gt = torch.as_tensor(gt, dtype=torch.long, device=pred_mask.device)
            if gt.ndim == 2:
                gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤

            # 🔥 0~255 → 0~1로 변환
            if gt.max() > 1:
                gt = (gt > 127).long()

            loss = torch.nn.functional.cross_entropy(pred_mask, gt)
            test_running_loss += loss.item() * gt.size(0)

            # IoU 계산
            preds = torch.softmax(pred_mask, dim=1).argmax(dim=1) 
            test_iou_list.append(binary_iou(preds, gt))

        test_mean_loss = test_running_loss / len(dataset_loader['testing'].dataset)
        test_miou = np.mean(test_iou_list) if test_iou_list else 0.0
        print(f'TEST || Loss - {test_mean_loss}, mIoU - {test_miou}')

        logger.log(epoch,
                    train_mean_loss, train_miou,
                    val_mean_loss, val_miou,
                    test_mean_loss, test_miou)

        if best_test_miou < test_miou:
            best_test_miou = test_miou

            ckpt = {
                "epoch": epoch,
                # "efficientsam": sam_model.state_dict(),
                "seg_decoder": seg_decoder.state_dict(),
                "adaptive_patch": adaptive_encoder.state_dict(),
                # "optimizer": optimizer.state_dict(),
                # "best_val_iou": val_best_iou,
            }
        
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(parent_dir, f"best_{model_type_name}_{date_time}.pth"))
            print(f" {model_type_name} -> New best mIoU {best_test_miou}. checkpoint saved.")
        


    print("Training done.")
    del sam_model
    del seg_decoder
    del adaptive_encoder
    del optimizer
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
    device = torch.device("cuda:0")

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
    date = datetime.now().strftime("%y%m%d")
    for model_type in model_types:

        print(f'Now Loaddig.... {model_type}') # 모델 타입에 따른 설정
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])
        seg_decoder = SegHead(sam_variant=model_type)
        adaptive_encoder = RODSegAdaptivePatch(model_type=model_type)
        
        

        train_orfd(
            dataset_root = image_file,
            sam_model=sam_model,
            adaptive_encoder = adaptive_encoder,
            seg_decoder=seg_decoder,
            device = device,
            epochs= 100,
            # batch_size=1,
            # num_workers=1,
            # val_every=2,
            model_type_name = model_type,
            date_time= date
        )

def write_train_log_csv(filepath: str, epoch: int, 
                        train_loss: float, train_miou: float, 
                        test_loss: float, test_miou: float,
                        val_loss: float, val_miou: float):
    """
    학습 결과를 CSV 형식으로 저장하는 함수.
    파일이 없으면 헤더 자동 생성.
    """
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 파일 없으면 헤더 작성
        if not file_exists:
            writer.writerow(["Epoch", "Train Loss", "Train mIoU", 
                             "Test Loss", "Test mIoU", 
                             "Val Loss", "Val mIoU"])

        writer.writerow([
            epoch, 
            round(train_loss, 6), round(train_miou, 6),
            round(test_loss, 6), round(test_miou, 6),
            round(val_loss, 6), round(val_miou, 6)
        ])

if __name__ == '__main__':
    main()