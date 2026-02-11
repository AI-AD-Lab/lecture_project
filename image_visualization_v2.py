import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from segment_anything import sam_model_registry
from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt

from seg_decoder import SegHead
from adaptive_encoding_patch_model_v2 import RODSegAdaptivePatch

from dataloader import ORFDDataset
import time

from train_utils import (
    make_dataloaders,
    binary_iou,
    postprocess_masks,
    preprocess,
    build_poly_scheduler,
    gt_processing,
)


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
    if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
        rgb_image = np.transpose(rgb_image, (1, 2, 0))

    overlay_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 3) 빈 컬러 마스크 생성
    mask_color = np.zeros_like(overlay_img, dtype=np.uint8)
    mask_color[pred == 1] = color  # 예측된 부분에 색칠

    # 4) 원본 이미지 위에 반투명 오버레이
    output = cv2.addWeighted(overlay_img, 1.0, mask_color, alpha, 0)

    # 5) 저장
    cv2.imwrite(str(save_path), output)


total_result_summary = {}

def main():

    model_types = [
        'vit_h',
        'vit_l', 
        'vit_b',
        'vits', 
        'vitt'
    ]
    
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

    # seghead_adaptive_patch_model_weight_chekpoint = {
    #     'vit_h': './ckpts_ap_canny/260122/vit_h_best_val.pth',
    #     'vit_l': './ckpts_ap_canny/260122/vit_l_best_val.pth',
    #     'vit_b': './ckpts_ap_canny/260122/vit_b_best_val.pth',
    #     'vits': './ckpts_ap_canny/260122_1/vits_best_val.pth',
    #     'vitt': './ckpts_ap_canny/260122/vitt_best_val.pth',  
    # }

    seghead_adaptive_patch_model_weight_chekpoint = {
        'vit_h': './ckpts_2/260121_canny_sensitive/vit_h_canny_sensitive_best_val.pth',
        'vit_l': './ckpts_2/260121_canny_sensitive/vit_l_canny_sensitive_best_val.pth',
        'vit_b': './ckpts_2/260121_canny_sensitive/vit_b_canny_sensitive_best_val.pth',
        'vits': './ckpts_2/260121_canny_sensitive/vits_canny_sensitive_best_val.pth',
        'vitt': './ckpts_2/260121_canny_sensitive/vitt_canny_sensitive_best_val.pth',  
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = './ORFD_dataset'


    # canny 파라미터(필요시만 튜닝)
    edge_mode = "canny"  # "sobel" or "canny"
    boundary_thresh = 0.0  # z-score threshold

    # defualt setting
    

    CANNY_PRESETS = {
        "canny_sensitive": dict(
            low_thr=0.08,
            high_thr=0.20,
            thr_sharpness=12.0,
            hyst_iters=3,
            boundary_thresh=-0.5,
        ),
        "canny_medium": dict(
            low_thr=0.12,
            high_thr=0.30,
            thr_sharpness=10.0,
            hyst_iters=2,
            boundary_thresh=0.0,
        ),
        "canny_default": dict(
            low_thr=0.10,
            high_thr=0.30,
            thr_sharpness=20.0,
            hyst_iters=2,
            boundary_thresh=0.0,
        ),
    }

    CANNY_KW = CANNY_PRESETS['canny_sensitive']

    for model_type in model_types:

        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type]).to(device).eval()
        seg_decoder = SegHead(sam_variant=model_type).to(device).eval()

        adaptive_encoder = RODSegAdaptivePatch(
            model_type=model_type,
            edge_mode=edge_mode,
            # boundary_thresh=boundary_thresh,
            **CANNY_KW,
        ).to(device).eval()

        best_model_path = seghead_adaptive_patch_model_weight_chekpoint[model_type]
        print("Loading checkpoint:", best_model_path)
        ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
        
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

        image_viz_canny(
            dataset_root= dataset_root,
            sam_model=sam_model,
            adaptive_encoder = adaptive_encoder,
            seg_decoder = seg_decoder,
            device = device,
            model_type_name=model_type,
            save_path_dir='code_2/output'
            # save_path_dir='output_ap_canny'
        )

    for model_t in total_result_summary.keys():
        each_result = total_result_summary[model_t]
        iou = each_result['IOU']
        f1 =  each_result['F1']

        print(f"{model_t}|| IOU: {iou:.4f}, F1: {f1:.4f}")



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



@torch.no_grad()
def image_viz_canny(
    dataset_root,
    sam_model,
    adaptive_encoder,
    seg_decoder,
    device,
    model_type_name='vits',
    save_path_dir='output_ap_canny'
):
    # 몇 장 저장할지
    batch_size = 2
    num_workers = 4

    
    ds = ORFDDataset(dataset_root, mode='testing')
    # print(f"[{split}] Loaded {len(ds)} image pairs.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # preprocess()가 1024x1024로 리사이즈하므로 input_size 고정으로 처리
    input_size = (1024, 1024)

    parent_dir = Path(save_path_dir) / model_type_name
    mask_image_save_dir = parent_dir / 'masking_img'
    masked_gt_image_dir = parent_dir / 'masked_img'

    path_list = [mask_image_save_dir, masked_gt_image_dir]
    for path_dir in path_list:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)

    running_loss = 0.0
    iou_list = []
    test_f1_list = []

    frame_start  = time.time()

    for imgs, gts, names in tqdm(loader, desc="viz"):
        # name = names[0] if isinstance(names, (list, tuple)) else str(names)

        # ---- imgs to GPU (배치 유지) ----
        if not torch.is_tensor(imgs):
            imgs = torch.as_tensor(imgs)
        imgs = _ensure_3ch_float_tensor(imgs)          # [B,3,H,W]
        imgs = imgs.to(device, non_blocking=True)

        B = imgs.shape[0]
        ori_size = (imgs.shape[-2], imgs.shape[-1])    # (H,W)

        input_image_torch = preprocess(imgs).to(device)  # [B,3,1024,1024]

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image_torch)

            ap_image_embedding = adaptive_encoder(input_image_torch, image_embedding)
            pred_mask = seg_decoder(ap_image_embedding)  # 기대: [B,2,256,256] 

            # postprocess -> 원본 해상도 [B,2,H,W]
            pred_mask = postprocess_masks(
                pred_mask,
                input_size=input_size,
                original_size=ori_size,
            )

            gt_tensor = _gt_to_BHW(gts, device=pred_mask.device)  # [B,H,W]
            loss = F.cross_entropy(pred_mask, gt_tensor)

            running_loss += loss.item() * B

        # IoU 계산 (binary_iou가 target [B,1,H,W] 기대면 unsqueeze)
        preds = torch.softmax(pred_mask, dim=1).argmax(dim=1)  # [B,H,W]
        iou_list.append(binary_iou(preds, gt_tensor.unsqueeze(1)))



        # ---- F1-score 계산 추가 ----
        f1 = binary_f1(preds, gt_tensor)
        test_f1_list.append(f1.item())

        for i in range(len(names)):
            dst_masking_image_save_path = mask_image_save_dir / names[i]
            dst_overlay_image_save_path = masked_gt_image_dir / names[i]

            overlayed_origin_img = np.array(imgs[i].cpu())
            # print(preds[i].shape)
            # save_pred_image(pred_mask[i], dst_masking_image_save_path)
            # save_overlay_image(overlayed_origin_img ,pred_mask[i], save_path = dst_overlay_image_save_path)
    
    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / max(dataset_size, 1)
    epoch_miou = float(np.mean(iou_list)) if iou_list else 0.0
    end_time = time.time()
    mean_test_f1 = sum(test_f1_list) / len(test_f1_list)

    total_time = (end_time - frame_start)
    avg_time = total_time / len(ds)
    fps = 1.0 / avg_time

    print(f'Testing Time: {total_time:.2f} seconds, FPS: {fps:.2f}')
    print(f'Test Loss: {epoch_loss:.4f}, Test mIoU: {epoch_miou:.4f}, Test F1-score: {mean_test_f1:.4f}')

    total_result_summary[model_type_name] = {'IOU':epoch_miou, "F1":mean_test_f1}

    # test_mean_loss = test_running_loss / len(ds)
    # test_miou = np.mean(test_iou_list) if test_iou_list else 0.0
    # print(f'TEST || Loss - {test_mean_loss}, mIoU - {test_miou}')

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

if __name__ == "__main__":
    main()
