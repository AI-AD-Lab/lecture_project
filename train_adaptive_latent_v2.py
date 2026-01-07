import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from segment_anything import sam_model_registry
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

from seg_decoder import SegHead
from adaptive_encoding_patch_model_v2 import RODSegAdaptivePatch

from train_utils import (
    make_dataloaders,
    binary_iou,
    postprocess_masks,
    preprocess,
    build_poly_scheduler,
    gt_processing,
)


def _ensure_3ch_float_tensor(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: [B,C,H,W] or [B,H,W] or [B,H,W,C] (가능하면 dataset에서 이미 [B,3,H,W]로 나오게 하는 게 최선)
    """
    if imgs.ndim == 3:
        # [B,H,W] -> [B,1,H,W]
        imgs = imgs.unsqueeze(1)

    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3, 4) and imgs.shape[1] not in (1, 3, 4):
        # [B,H,W,C] -> [B,C,H,W]
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # channel 정리
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    elif imgs.shape[1] == 4:
        imgs = imgs[:, :3, :, :]

    return imgs


def _gt_to_BHW(gts: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    gts를 CE용 [B,H,W] long(0/1)로 변환
    - gts가 [B,1,H,W] or [B,H,W] or [H,W] 등으로 와도 처리
    """
    if not torch.is_tensor(gts):
        gts = torch.as_tensor(gts)

    gts = gts.to(device, non_blocking=True)

    # (H,W) -> (1,H,W)
    if gts.ndim == 2:
        gts = gts.unsqueeze(0)

    # (B,1,H,W) -> (B,H,W)
    if gts.ndim == 4:
        gts = gts.squeeze(1)

    if gts.ndim != 3:
        raise ValueError(f"Unexpected gts shape: {tuple(gts.shape)}")

    # 0~255면 0/1로
    if gts.max() > 1:
        gts = (gts > 128).long()
    else:
        gts = gts.long()

    return gts


def train_orfd(
    dataset_root: str,
    sam_model,
    adaptive_encoder: nn.Module,
    seg_decoder: nn.Module,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    model_type_name: str = "vits",
    parent_dir: str = "ckpts_adaptive",
    date_time: str = "250000",
    patience: int = 5,
    batch_size: int = 4,
    num_workers: int = 4,
    use_amp: bool = True,
):
    phases = ["training", "validation", "testing"]

    # Dataset / Loader
    dataloaders = make_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 고정 SAM backbone
    sam_model.to(device).eval()
    adaptive_encoder.to(device)
    seg_decoder.to(device)

    # adaptive_encoder, seg_decoder만 학습
    optimizer = torch.optim.AdamW(
        [
            {"params": seg_decoder.parameters(), "lr": lr},
            {"params": adaptive_encoder.parameters(), "lr": lr},
        ],
        weight_decay=1e-3,
    )

    total_steps = len(dataloaders["training"]) * epochs
    scheduler = build_poly_scheduler(optimizer, total_steps, power=0.9)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Train {model_type_name} | batch={batch_size}, workers={num_workers}, amp={use_amp}")

    parent_dir_with_date = Path(parent_dir) / date_time
    os.makedirs(parent_dir_with_date, exist_ok=True)
    log_path = os.path.join(parent_dir_with_date, f"{model_type_name}_train_log.txt")

    def log_write(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    best_val_miou = 0.0
    best_test_miou = 0.0
    early_stop_count = 0
    global_step = 0

    # preprocess()가 1024x1024로 맞추므로 input_size는 고정 취급 가능
    input_size: Tuple[int, int] = (1024, 1024)

    for epoch in range(1, epochs + 1):
        log_write(f"{model_type_name} || ===== Epoch {epoch}/{epochs} =====")

        for phase in phases:
            if phase not in dataloaders:
                continue

            is_train = (phase == "training")

            if is_train:
                seg_decoder.train()
                adaptive_encoder.train()
            else:
                seg_decoder.eval()
                adaptive_encoder.eval()

            running_loss = 0.0
            iou_list = []
            dataloader = dataloaders[phase]

            for imgs, gts, _ in tqdm(dataloader, desc=phase):
                # ---- imgs to GPU (배치 유지) ----
                if not torch.is_tensor(imgs):
                    imgs = torch.as_tensor(imgs)
                imgs = _ensure_3ch_float_tensor(imgs)          # [B,3,H,W]
                imgs = imgs.to(device, non_blocking=True)

                B = imgs.shape[0]
                ori_size = (imgs.shape[-2], imgs.shape[-1])    # (H,W)

                # preprocess: normalize + (1024,1024) resize (배치 처리)
                input_image_torch = preprocess(imgs).to(device)  # [B,3,1024,1024]

                # ---- SAM backbone (frozen) ----
                with torch.no_grad():
                    # 일부 efficient_sam 구현은 image_encoder API가 다를 수 있음
                    # 너 프로젝트에서 sam_model.image_encoder(input) 이 동작하므로 그대로 사용
                    image_embedding = sam_model.image_encoder(input_image_torch)

                # ---- forward / loss ----
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        ap_image_embedding = adaptive_encoder(input_image_torch, image_embedding)
                        pred_mask = seg_decoder(ap_image_embedding)  # 기대: [B,2,256,256] 등

                        # postprocess -> 원본 해상도 [B,2,H,W]
                        pred_mask = postprocess_masks(
                            pred_mask,
                            input_size=input_size,
                            original_size=ori_size,
                        )

                        gt_tensor = _gt_to_BHW(gts, device=pred_mask.device)  # [B,H,W]
                        loss = F.cross_entropy(pred_mask, gt_tensor)

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        global_step += 1

                running_loss += loss.item() * B

                # IoU 계산 (binary_iou가 target [B,1,H,W] 기대면 unsqueeze)
                preds = torch.softmax(pred_mask, dim=1).argmax(dim=1)  # [B,H,W]
                iou_list.append(binary_iou(preds, gt_tensor.unsqueeze(1)))

            dataset_size = len(dataloader.dataset)
            epoch_loss = running_loss / max(dataset_size, 1)
            epoch_miou = float(np.mean(iou_list)) if iou_list else 0.0

            log_write(f"{phase.upper()} || Loss: {epoch_loss:.4f}, mIoU: {epoch_miou:.4f}")

            # validation 기준 best 저장 + early stopping
            if phase == "validation":
                if epoch_miou > best_val_miou:
                    best_val_miou = epoch_miou
                    early_stop_count = 0

                    ckpt_path = os.path.join(
                        parent_dir_with_date,
                        f"{model_type_name}_best_val.pth",
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_type_name": model_type_name,
                            "adaptive_encoder": adaptive_encoder.state_dict(),
                            "seg_decoder": seg_decoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_val_miou": best_val_miou,
                            "batch_size": batch_size,
                            "num_workers": num_workers,
                            "use_amp": use_amp,
                        },
                        ckpt_path,
                    )
                    log_write(f">> New best validation mIoU: {best_val_miou:.4f} (saved to {ckpt_path})")
                else:
                    early_stop_count += 1
                    log_write(f"No improvement on validation. Early stop counter: {early_stop_count}/{patience}")

            # test 최고 기록 (저장은 validation best만)
            if phase == "testing":
                if epoch_miou > best_test_miou:
                    best_test_miou = epoch_miou

        if early_stop_count >= patience:
            log_write(f"Early stopping triggered at epoch {epoch}")
            break

    log_write(f"Training finished. Best val mIoU: {best_val_miou:.4f}, Best test mIoU: {best_test_miou:.4f}")

    return {"best_val_miou": best_val_miou, "best_test_miou": best_test_miou}


def main():
    model_types = [
        # "vit_h",
        "vit_l",
        "vit_b",
        "vits",
        # "vitt",
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

    dataset_root = "./ORFD_dataset"
    date = datetime.now().strftime("%y%m%d")

    for model_type in model_types:
        print(f"Now Loading.... {model_type}")
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])

        seg_decoder = SegHead(sam_variant=model_type)
        adaptive_encoder = RODSegAdaptivePatch(model_type=model_type)

        train_orfd(
            dataset_root=dataset_root,
            sam_model=sam_model,
            adaptive_encoder=adaptive_encoder,
            seg_decoder=seg_decoder,
            device=device,
            epochs=20,
            lr=1e-4,
            parent_dir="ckpts_adaptive",
            model_type_name=model_type,
            date_time=date,
            patience=5,
            batch_size=4,
            num_workers=4,
            use_amp=True,
        )


if __name__ == "__main__":
    main()
