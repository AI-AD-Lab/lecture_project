# image_visualization_0107.py
# ------------------------------------------------------------
# 목적:
#  1) best_val checkpoint 로드
#  2) validation 데이터에서 1~N장 inference
#  3) 아래 5가지를 파일로 저장
#     - RGB 원본
#     - GT (binary)
#     - Pred mask (binary)
#     - Overlay (RGB + Pred)
#     - Boundary mask (patch grid + upsampled)
#  4) edge_mode="canny" / "sobel" 스위치 가능
# ------------------------------------------------------------

import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt

from seg_decoder import SegHead
from adaptive_encoding_patch_model_v2 import RODSegAdaptivePatch

from dataloader import ORFDDataset
from train_utils import preprocess, postprocess_masks


def _ensure_3ch_chw(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: [B,3,H,W] or [B,H,W,3] or [B,1,H,W] etc.
    return: [B,3,H,W] uint8/float 상관 없음
    """
    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(imgs)

    if imgs.ndim == 3:
        # [B,H,W] -> [B,1,H,W]
        imgs = imgs.unsqueeze(1)

    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3, 4) and imgs.shape[1] not in (1, 3, 4):
        # [B,H,W,C] -> [B,C,H,W]
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    elif imgs.shape[1] == 4:
        imgs = imgs[:, :3]

    return imgs


def _gt_to_hw01(gts) -> np.ndarray:
    """
    gts: tensor/np, shape could be [1,H,W], [H,W], [1,1,H,W], ...
    return: (H,W) uint8 0/255
    """
    if torch.is_tensor(gts):
        g = gts.detach().cpu().numpy()
    else:
        g = np.asarray(gts)

    # squeeze
    while g.ndim > 2:
        g = g.squeeze(0)

    # 0~255 or 0/1 -> 0/255
    if g.max() > 1:
        g01 = (g > 128).astype(np.uint8)
    else:
        g01 = (g > 0.5).astype(np.uint8)

    return g01 * 255


def _to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: [3,H,W] tensor on CPU/GPU
    return: (H,W,3) uint8 in RGB order
    """
    img = img_chw.detach().cpu()
    if img.dtype != torch.uint8:
        # 데이터셋이 이미 0~255 float일 가능성이 큼. 안전 처리.
        img = img.clamp(0, 255).to(torch.uint8)
    rgb = img.permute(1, 2, 0).numpy()  # HWC, RGB
    return rgb


def _save_overlay(rgb: np.ndarray, pred01_255: np.ndarray, out_path: str, alpha: float = 0.5):
    """
    rgb: (H,W,3) uint8 RGB
    pred01_255: (H,W) uint8 0/255
    overlay: pred=1 영역을 빨간색으로
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = np.zeros_like(bgr)
    color[pred01_255 > 0] = (0, 0, 255)  # red in BGR
    out = cv2.addWeighted(bgr, 1.0, color, alpha, 0.0)
    cv2.imwrite(out_path, out)


@torch.no_grad()
def main():
    # -----------------------
    # [사용자 설정]
    # -----------------------
    model_type = "vit_l"  # "vit_b" or "vits" 등
    best_model_path = r"ckpts_adaptive/260107/vit_l_best_val.pth"  # 너 파일명에 맞게
    dataset_root = "./ORFD_dataset"
    split = "testing"  # "testing"로 바꿔도 됨
    out_dir = Path(f"viz_outputs/{model_type}_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # canny/sobel 스위치
    edge_mode = "canny"  # "sobel" or "canny"
    boundary_thresh = 0.0  # z-score threshold

    # canny 파라미터(필요시만 튜닝)
    low_thr = 0.10
    high_thr = 0.30
    thr_sharpness = 20.0
    hyst_iters = 2

    # 몇 장 저장할지
    max_save = 90000

    # batch는 시각화면 1 추천(헷갈림 방지)
    batch_size = 1
    num_workers = 1

    # -----------------------
    # device
    # -----------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # -----------------------
    # SAM backbone 로드
    # -----------------------
    sam_model_dict = {
        "vit_h": sam_model_registry["vit_h"],
        "vit_l": sam_model_registry["vit_l"],
        "vit_b": sam_model_registry["vit_b"],
        "vits": build_efficient_sam_vits,
        "vitt": build_efficient_sam_vitt,
    }

    sam_weights = {
        "vit_h": "./weights/sam_vit_h_4b8939.pth",
        "vit_l": "./weights/sam_vit_l_0b3195.pth",
        "vit_b": "./weights/sam_vit_b_01ec64.pth",
        "vits": "./weights/efficient_sam_vits.pt",
        "vitt": "./weights/efficient_sam_vitt.pt",
    }

    print("Loading SAM:", model_type)
    sam_model = sam_model_dict[model_type](checkpoint=sam_weights[model_type]).to(device).eval()

    # -----------------------
    # Head + Adaptive encoder 로드
    # -----------------------
    seg_decoder = SegHead(sam_variant=model_type).to(device).eval()

    adaptive_encoder = RODSegAdaptivePatch(
        model_type=model_type,
        edge_mode=edge_mode,
        boundary_thresh=boundary_thresh,
        low_thr=low_thr,
        high_thr=high_thr,
        thr_sharpness=thr_sharpness,
        hyst_iters=hyst_iters,
    ).to(device).eval()

    print("Loading checkpoint:", best_model_path)
    ckpt = torch.load(best_model_path, map_location="cpu")

    # 네 학습 코드 저장 구조: ckpt["adaptive_encoder"], ckpt["seg_decoder"]
    if "adaptive_encoder" in ckpt and "seg_decoder" in ckpt:
        adaptive_encoder.load_state_dict(ckpt["adaptive_encoder"], strict=True)
        seg_decoder.load_state_dict(ckpt["seg_decoder"], strict=True)
    else:
        raise KeyError(
            "Checkpoint에 'adaptive_encoder'/'seg_decoder' 키가 없음. "
            "train 저장 형식 확인 필요."
        )

    adaptive_encoder.to(device).eval()
    seg_decoder.to(device).eval()

    # -----------------------
    # dataset
    # -----------------------
    ds = ORFDDataset(dataset_root, mode=split)
    print(f"[{split}] Loaded {len(ds)} image pairs.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # preprocess()가 1024x1024로 리사이즈하므로 input_size 고정으로 처리
    input_size = (1024, 1024)

    saved = 0
    for imgs, gts, names in tqdm(loader, desc="viz"):
        if saved >= max_save:
            break

        name = names[0] if isinstance(names, (list, tuple)) else str(names)

        # ---- imgs: to GPU ----
        imgs = _ensure_3ch_chw(imgs).float()  # [B,3,H,W]
        # 원본 사이즈(저장/overlay용)
        ori_h, ori_w = int(imgs.shape[-2]), int(imgs.shape[-1])

        imgs = imgs.to(device, non_blocking=True)

        # ---- preprocess (normalize + resize 1024) ----
        x = preprocess(imgs)  # 보통 CPU 텐서로 처리할 수도 있으니
        x = x.to(device, non_blocking=True)  # ★ 이게 없으면 너가 본 에러 뜸

        # ---- SAM embedding ----
        image_embedding = sam_model.image_encoder(x)

        # adaptive_encoder가 (x,out) 형식을 기대하면 감싸기
        feature_in = image_embedding if isinstance(image_embedding, (tuple, list)) else (image_embedding, None)

        # ---- boundary analysis ----
        boundary_mask, s16, s8, grad = adaptive_encoder.boundary_analysis(x, feature_in)
        # boundary_mask: [B,1,64,64] 같은 patch grid
        bm = boundary_mask[0, 0].detach().float().cpu().numpy()  # 0/1

        # 보기 좋게 원본 크기로 upsample (nearest)
        bm_up = cv2.resize(bm, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        bm_up_255 = (bm_up * 255).astype(np.uint8)

        # ---- forward ----
        ap_feat = adaptive_encoder(x, feature_in)
        logits = seg_decoder(ap_feat)  # [B,2,256,256] 예상
        logits = postprocess_masks(logits, input_size=input_size, original_size=(ori_h, ori_w))  # [B,2,H,W]

        pred = torch.softmax(logits, dim=1).argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        pred_255 = pred * 255

        gt_255 = _gt_to_hw01(gts[0])

        # ---- save ----
        rgb = _to_uint8_rgb(imgs[0])  # 원본 이미지(전처리 전) 저장용

        cv2.imwrite(str(out_dir / f"{name}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{name}_gt.png"), gt_255)
        cv2.imwrite(str(out_dir / f"{name}_pred.png"), pred_255)
        cv2.imwrite(str(out_dir / f"{name}_boundary.png"), bm_up_255)
        _save_overlay(rgb, pred_255, str(out_dir / f"{name}_overlay.png"), alpha=0.5)

        saved += 1

    print(f"Done. Saved {saved} samples to: {out_dir.resolve()}")
    print(f"edge_mode = {edge_mode}")


if __name__ == "__main__":
    main()
