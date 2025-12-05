import cv2  # type: ignore
from segment_anything import sam_model_registry
from tqdm import tqdm

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

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from torchvision import transforms
from adaptive_encoding_patch_model import RODSegAdaptivePatch
from dataloader import ORFDDataset
from torch.utils.data import DataLoader


def build_poly_scheduler(optimizer, total_steps, power=0.9):
    def poly_decay(step):
        step = min(step, total_steps)
        return (1 - step / float(total_steps)) ** power
    return LambdaLR(optimizer, lr_lambda=poly_decay)


def preprocess(x):
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


@torch.no_grad()
def binary_iou(pred_mask, target):
    """
    pred_mask: [B,H,W] (0/1)
    target   : [B,1,H,W] (0/1)
    """
    # pred_mask: [B,H,W], target: [B,1,H,W]
    pred_mask = pred_mask.bool()
    target = target.bool()

    inter = (pred_mask & target.squeeze(1)).sum(dim=(1, 2)).float()
    union = (pred_mask | target.squeeze(1)).sum(dim=(1, 2)).float().clamp_min(1.0)
    return (inter / union).mean().item()


def make_dataloaders(dataset_root, batch_size=1, num_workers=1):
    phases = ["training", "testing", "validation"]
    datasets = {phase: ORFDDataset(dataset_root, mode=phase) for phase in phases}
    loaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=batch_size,
            shuffle=(phase == "training"),
            num_workers=num_workers,
            drop_last=(phase == "training"),
        )
        for phase in phases
    }
    return loaders

def forward_one_batch(
    imgs,
    gts,
    sam_model,
    adaptive_encoder,
    seg_decoder,
    device,
    transform,
):
    """
    imgs: [B,3,H,W] (torch.Tensor)
    gts : [B,H,W]   (torch.Tensor, 0/255 or 0/1)
    """
    imgs = imgs.to(device)
    gts = gts.to(device)
    B = imgs.shape[0]

    rgb_np_list = []
    ori_sizes = []
    input_sizes = []

    # 1) ResizeLongestSide 를 sample 단위로 적용
    for b in range(B):
        rgb = imgs[b].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
        ori_size = rgb.shape[:2]            # (H, W)
        input_image = transform.apply_image(rgb)  # resize
        input_size = input_image.shape[:2]  # (H', W')

        rgb_np_list.append(input_image)
        ori_sizes.append(ori_size)
        input_sizes.append(input_size)

    # 2) 다시 텐서로 묶기: [B,3,H',W']
    input_image_torch = torch.stack(
        [
            torch.as_tensor(img).permute(2, 0, 1).contiguous()
            for img in rgb_np_list
        ],
        dim=0,
    )  # [B,3,H',W']

    # 3) SAM 스타일 전처리
    input_image_torch = preprocess(input_image_torch).to(device)

    # 4) GT 전처리
    # gts: [B,H,W] 가정
    if gts.ndim == 3:
        gt = gts  # [B,H,W]
    elif gts.ndim == 2:
        gt = gts.unsqueeze(0)  # [1,H,W]
    else:
        raise ValueError(f"Unexpected gts shape: {gts.shape}")

    if gt.max() > 1:
        gt = (gt > 127).long()  # 0/255 → 0/1

    # 5) SAM encoder는 gradient 없이
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(input_image_torch)
        # image_embedding: EfficientSAM 구조에 따라 [B,C,H16,W16] or list/tuple

    # 6) Adaptive encoder + decoder (학습 대상)
    ap_image_embedding = adaptive_encoder(input_image_torch, image_embedding)
    pred_mask = seg_decoder(ap_image_embedding)  # [B,2,h',w'] (예: [B,2,256,256])

    # 7) sample별로 postprocess_masks 적용 (ori_size, input_size 다를 수 있음)
    pred_list = []
    for b in range(B):
        pm_b = pred_mask[b : b + 1]  # [1,2,h',w']
        pm_b = postprocess_masks(pm_b, input_sizes[b], ori_sizes[b])  # [1,2,H,W]
        pred_list.append(pm_b)

    pred_mask_full = torch.cat(pred_list, dim=0)  # [B,2,H,W]

    # 8) GT와 spatial size 맞추기 (혹시라도 다를 수 있으면)
    if pred_mask_full.shape[-2:] != gt.shape[-2:]:
        pred_mask_full = F.interpolate(
            pred_mask_full,
            size=gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    # 9) CE loss
    loss = F.cross_entropy(pred_mask_full, gt)  # [B,2,H,W] vs [B,H,W]

    # 10) IoU 계산
    preds = torch.softmax(pred_mask_full, dim=1).argmax(dim=1)  # [B,H,W]
    iou = binary_iou(preds, gt.unsqueeze(1))  # target [B,1,H,W]

    return loss, iou

def run_phase(
    phase,
    dataloader,
    sam_model,
    adaptive_encoder,
    seg_decoder,
    device,
    optimizer,
    scheduler,
    transform,
):
    is_train = phase == "training"

    if is_train:
        seg_decoder.train()
        adaptive_encoder.train()
    else:
        seg_decoder.eval()
        adaptive_encoder.eval()

    running_loss = 0.0
    iou_list = []

    loop = tqdm(dataloader, desc=phase.upper())
    for imgs, gts, _ in loop:
        if is_train:
            optimizer.zero_grad()
            loss, iou = forward_one_batch(
                imgs, gts,
                sam_model, adaptive_encoder, seg_decoder,
                device, transform
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            with torch.no_grad():
                loss, iou = forward_one_batch(
                    imgs, gts,
                    sam_model, adaptive_encoder, seg_decoder,
                    device, transform
                )

        running_loss += loss.item() * imgs.size(0)
        iou_list.append(iou)
        loop.set_postfix(loss=loss.item(), iou=iou)

    mean_loss = running_loss / len(dataloader.dataset)
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0

    return mean_loss, mean_iou

def train_orfd(
    dataset_root,
    sam_model,
    adaptive_encoder,
    seg_decoder,
    device,
    epochs=100,
    lr=1e-4,
    model_type_name="vits",
    early_stop_patience=10,
    batch_size=4,
    num_workers=4,
):

    loaders = make_dataloaders(dataset_root, batch_size=batch_size, num_workers=num_workers)

    sam_model.to(device).eval()  # encoder freeze
    adaptive_encoder.to(device)
    seg_decoder.to(device)

    optimizer = AdamW(
        [
            {"params": seg_decoder.parameters(), "lr": lr},
            {"params": adaptive_encoder.parameters(), "lr": lr * 0.5},
        ],
        weight_decay=1e-3,
    )

    total_steps = len(loaders["training"]) * epochs
    scheduler = build_poly_scheduler(optimizer, total_steps, power=0.9)
    transform = ResizeLongestSide(1024)

    best_val_miou = 0.0
    best_test_miou = 0.0
    early_stop_count = 0

    print(f"Train {model_type_name} (batch_size={batch_size})")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_miou = run_phase(
            "training",
            loaders["training"],
            sam_model,
            adaptive_encoder,
            seg_decoder,
            device,
            optimizer,
            scheduler,
            transform,
        )
        print(f"TRAIN || Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")

        test_loss, test_miou = run_phase(
            "testing",
            loaders["testing"],
            sam_model,
            adaptive_encoder,
            seg_decoder,
            device,
            optimizer,
            scheduler,
            transform,
        )
        print(f"TEST  || Loss: {test_loss:.4f}, mIoU: {test_miou:.4f}")

        if test_miou <= best_test_miou:
            early_stop_count += 1
            print(f"No test mIoU improvement for {early_stop_count} steps...")
            if early_stop_count >= early_stop_patience:
                print("Early stop activated.")
                break
        else:
            best_test_miou = test_miou
            early_stop_count = 0

        val_loss, val_miou = run_phase(
            "validation",
            loaders["validation"],
            sam_model,
            adaptive_encoder,
            seg_decoder,
            device,
            optimizer,
            scheduler,
            transform,
        )
        print(f"VAL   || Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            ckpt = {
                "epoch": epoch,
                "seg_decoder": seg_decoder.state_dict(),
                "adaptive_patch": adaptive_encoder.state_dict(),
                "best_val_miou": best_val_miou,
            }
            parent_dir = "ckpts_adaptive_opt"
            os.makedirs(parent_dir, exist_ok=True)
            save_path = os.path.join(parent_dir, f"best_{model_type_name}_251204.pth")
            torch.save(ckpt, save_path)
            print(f"{model_type_name} -> New best mIoU {best_val_miou:.4f}. checkpoint saved at {save_path}")

    print("Training done.")
    del sam_model, seg_decoder, adaptive_encoder, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")


def main():
    model_types = [
        # 'vit_h',
        # 'vit_l', 
        # 'vit_b',
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

    for model_type in model_types:
        print(f"Now Loading.... {model_type}")
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])
        seg_decoder = SegHead(sam_variant=model_type)
        adaptive_encoder = RODSegAdaptivePatch(model_type=model_type)

        train_orfd(
            dataset_root=image_file,
            sam_model=sam_model,
            adaptive_encoder=adaptive_encoder,
            seg_decoder=seg_decoder,
            device=device,
            epochs=30,
            lr=1e-4,
            model_type_name=model_type,
        )


if __name__ == "__main__":
    main()
