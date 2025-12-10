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

def gt_processing(gt, device):
    # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
    gt = torch.as_tensor(gt, dtype=torch.long, device=device)
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤

    # 🔥 0~255 → 0~1로 변환
    if gt.max() > 1:
        gt = (gt > 128).long()
    return gt


def train_orfd(
    dataset_root,
    sam_model,
    seg_decoder,
    device,
    epochs=100,
    lr=1e-4,
    model_type_name='vits',
    parent_dir = 'ckpts_seghead',
    date_time='250000',
    patience=20,          # early stopping
):
    phases = ['training', 'validation', 'testing']

    # Dataset / Loader, **"ResizeLongestSide 때문에 배치가 높아도 1개씩만 수행해야 함"**
    dataloaders = make_dataloaders(
        dataset_root=dataset_root,
        batch_size=1,
        num_workers=1
    )

    # 고정 SAM backbone
    sam_model.to(device).eval()
    seg_decoder.to(device)

    #  seg_decoder만 학습
    optimizer = torch.optim.AdamW(
        [
            {"params": seg_decoder.parameters(), "lr": lr},
        ],
        weight_decay=1e-3,
    )

    total_steps = len(dataloaders['training']) * epochs
    scheduler = build_poly_scheduler(optimizer, total_steps, power=0.9)
    transform = ResizeLongestSide(1024)

    print(f'Train {model_type_name}')

    parent_dir_with_date = Path(parent_dir) / date_time
    os.makedirs(parent_dir_with_date, exist_ok=True)
    log_path = os.path.join(parent_dir_with_date, f"{model_type_name}_train_log.txt")

    def log_write(msg: str):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    best_val_miou = 0.0
    best_test_miou = 0.0
    early_stop_count = 0

    global_step = 0

    for epoch in range(1, epochs + 1):
        log_write(f'===== Epoch {epoch}/{epochs} =====')

        # 각 phase 별로 공통 루프 사용
        for phase in phases:
            if phase not in dataloaders:
                continue

            is_train = (phase == 'training')

            if is_train:
                seg_decoder.train()
            else:
                seg_decoder.eval()

            running_loss = 0.0
            iou_list = []

            dataloader = dataloaders[phase]

            for imgs, gts, _ in tqdm(dataloader, desc=phase):
                # 이미지 하나씩 처리 (ResizeLongestSide 때문에), sam 모델 전처리 떄문
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

                # train / val / test 모두에서 forward는 동일
                with torch.set_grad_enabled(is_train):
                    pred_mask = seg_decoder(image_embedding)       # [1, 2, 256, 256]
                    pred_mask = postprocess_masks(                    # [1, 2, H, W]
                        pred_mask, input_size, ori_size
                    )

                    # GT 전처리
                    gt_tensor = gt_processing(gt, device=pred_mask.device)

                    # cross entropy loss
                    loss = F.cross_entropy(pred_mask, gt_tensor)

                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        global_step += 1

                batch_size = gt_tensor.size(0)
                running_loss += loss.item() * batch_size

                # IoU 계산
                preds = torch.softmax(pred_mask, dim=1).argmax(dim=1)
                iou_list.append(binary_iou(preds, gt_tensor))

            dataset_size = len(dataloader.dataset)
            epoch_loss = running_loss / max(dataset_size, 1)
            epoch_miou = float(np.mean(iou_list)) if iou_list else 0.0

            log_write(f'{phase.upper()} || Loss: {epoch_loss:.4f}, mIoU: {epoch_miou:.4f}')

            # validation 기준으로 best model 및 early stopping
            if phase == 'validation':
                if epoch_miou > best_val_miou:
                    best_val_miou = epoch_miou
                    early_stop_count = 0

                    # best validation 모델 저장
                    ckpt_path = os.path.join(
                        parent_dir_with_date,
                        f'{model_type_name}_best_val_{date_time}.pth'
                    )
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_type_name': model_type_name,
                            'seg_decoder': seg_decoder.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_val_miou': best_val_miou,
                        },
                        ckpt_path,
                    )
                    log_write(f'>> New best validation mIoU: {best_val_miou:.4f} (saved to {ckpt_path})')
                else:
                    early_stop_count += 1
                    log_write(f'No improvement on validation. Early stop counter: {early_stop_count}/{patience}')

            # test 성능도 최고 값 기록 (모델 저장은 validation 기준)
            if phase == 'testing':
                if epoch_miou > best_test_miou:
                    best_test_miou = epoch_miou

        # epoch 끝나고 early stopping 체크
        if early_stop_count >= patience:
            log_write(f'Early stopping triggered at epoch {epoch}')
            break

    log_write(f'Training finished. Best val mIoU: {best_val_miou:.4f}, Best test mIoU: {best_test_miou:.4f}')

    return {
        'best_val_miou': best_val_miou,
        'best_test_miou': best_test_miou,
    }

    
def main():
    model_types = [
        'vit_h',
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
    date = datetime.now().strftime("%y%m%d")

    for model_type in model_types:
        print(f"Now Loading.... {model_type}")
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])
        seg_decoder = SegHead(sam_variant=model_type)

        train_orfd(
            dataset_root=image_file,
            sam_model=sam_model,
            seg_decoder=seg_decoder,
            device=device,
            epochs=100,
            lr=1e-4,
            parent_dir = f'ckpts_seghead',
            model_type_name=model_type,
            date_time = date, 
            patience=20
        )


if __name__ == "__main__":
    main()
