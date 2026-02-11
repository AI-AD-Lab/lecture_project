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
from adaptive_encoding_patch_model_v2 import RODSegAdaptivePatch as sobel_ap
from adaptive_encoding_patch_model_v2 import RODSegAdaptivePatch as canny_ap

from dataloader import ORFDDataset
import time

from train_utils import (
    make_dataloaders,
    # binary_iou,
    postprocess_masks,
    preprocess,
    build_poly_scheduler,
    gt_processing,
)

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

@torch.no_grad()
def binary_iou(pred_mask, target):
    """
    pred_mask: [B, W, H] (0/1)
    target  : [B, W, H] (0/1)
    return  : List[float] length = B
    """
    pred_mask = pred_mask.bool()
    target = target.bool()
    target = target.squeeze(1)

    # batch별 intersection / union
    inter = (pred_mask & target).sum(dim=(1, 2)).float()
    union = (pred_mask | target).sum(dim=(1, 2)).float().clamp_min(1.0)

    iou = inter / union          # [B]

    return iou.tolist() 

@torch.no_grad()
def main():

    model_types = [
        'vit_h',
        'vit_l', 
        'vit_b',
        'vits', 
        'vitt'
    ]

    # sam 모델은 변경사항 없음
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

    # basic seghead model weight path
    chkpt_seghead = {
        'vit_h': './ckpts_2/260126_fixed/vit_h_canny_sensitive_best_val.pth',
        'vit_l': './ckpts_2/260126_fixed/vit_l_canny_sensitive_best_val.pth',
        'vit_b': './ckpts_2/260126_fixed/vit_b_canny_sensitive_best_val.pth',
        'vits': './ckpts_2/260126_fixed/vits_canny_sensitive_best_val.pth',
        'vitt': './ckpts_2/260126_fixed/vitt_canny_sensitive_best_val.pth',  
    }

    # Sobel ap patch model weight path
    chkpt_sobel_ap = {
        'vit_h': './ckpts_2/260112/vit_h_best_val.pth',
        'vit_l': './ckpts_2/260112/vit_l_best_val.pth',
        'vit_b': './ckpts_2/260112/vit_b_best_val.pth',
        'vits': './ckpts_2/260112/vits_best_val.pth',
        'vitt': './ckpts_2/260112/vitt_best_val.pth',  
    }

    # Canny ap patch model weight
    # chkpt_canny_ap = {
    #     'vit_h': './ckpts_2/260108/vit_h_best_val.pth',
    #     'vit_l': './ckpts_2/260108/vit_l_best_val.pth',
    #     'vit_b': './ckpts_2/260108/vit_b_best_val.pth',
    #     'vits': './ckpts_2/260108/vits_best_val.pth',
    #     'vitt': './ckpts_2/260108/vitt_best_val.pth',  
    # }

    chkpt_canny_ap = {
    'vit_h': './ckpts_2/260121_canny_default/vit_h_canny_default_best_val.pth',
    'vit_l': './ckpts_2/260121_canny_default/vit_l_canny_default_best_val.pth',
    'vit_b': './ckpts_2/260121_canny_default/vit_b_canny_default_best_val.pth',
    'vits': './ckpts_2/260121_canny_default/vits_canny_default_best_val.pth',
    'vitt': './ckpts_2/260121_canny_default/vitt_canny_default_best_val.pth',  
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = './ORFD_dataset'
    save_path_dir = "./output_comparing_iou"

    # canny ap model 파라미터
    edge_mode = "canny"  # "sobel" or "canny"
    boundary_thresh = 0.0  # z-score threshold
    low_thr = 0.10
    high_thr = 0.30
    thr_sharpness = 20.0
    hyst_iters = 2

    for model_type in model_types:

        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type]).to(device).eval()

        seg_decoder = SegHead(sam_variant=model_type).to(device).eval()        
        seg_decoder_sobel = SegHead(sam_variant=model_type).to(device).eval()
        seg_decoder_canny = SegHead(sam_variant=model_type).to(device).eval()
        
        ap_encoder_sobel = sobel_ap(model_type=model_type)
        ap_encoder_canny = canny_ap(
            model_type=model_type,
            edge_mode=edge_mode,
            boundary_thresh=boundary_thresh,
            low_thr=low_thr,
            high_thr=high_thr,
            thr_sharpness=thr_sharpness,
            hyst_iters=hyst_iters,
        ).to(device).eval()

        ckpt_seghead_path = chkpt_seghead[model_type]
        ckpt_sobel_path = chkpt_sobel_ap[model_type]
        ckpt_canny_path = chkpt_canny_ap[model_type]

        # model weight loading
        print(f"Loading checkpoint: \n seghead: {ckpt_seghead_path} \n sobel:{ckpt_sobel_path}, \n canny:{ckpt_canny_path} " )
        ckpt_seghead_weight = torch.load(ckpt_seghead_path, weights_only=False)
        ckpt_sobel_weight = torch.load(ckpt_sobel_path, weights_only=False)
        ckpt_canny_weight = torch.load(ckpt_canny_path, weights_only=False)

        seg_decoder.load_state_dict(ckpt_seghead_weight['seg_decoder'], strict=True)
        ap_encoder_sobel.load_state_dict(ckpt_sobel_weight["adaptive_encoder"], strict=True)
        seg_decoder_sobel.load_state_dict(ckpt_sobel_weight["seg_decoder"], strict=True)

        ap_encoder_canny.load_state_dict(ckpt_canny_weight["adaptive_encoder"], strict=True)
        seg_decoder_canny.load_state_dict(ckpt_canny_weight["seg_decoder"], strict=True)
        

        # print(ckpt_seghead_weight.keys())
        # continue
        # model deivce 
        sam_model.to(device)
        ap_encoder_sobel.to(device)
        ap_encoder_canny.to(device)
        seg_decoder.to(device)
        seg_decoder_sobel.to(device)
        seg_decoder_canny.to(device)

        # 몇 장 저장할지
        batch_size = 2
        num_workers = 4
        dataset = ORFDDataset(dataset_root, mode='testing')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # preprocess()가 1024x1024로 리사이즈하므로 input_size 고정으로 처리
        input_size = (1024, 1024)

        parent_dir = Path(save_path_dir) / model_type

        seghead_masekd_img_dir = parent_dir / 'seghead_masked_img'
        sobel_masekd_img_dir = parent_dir / 'sobel_masked_img'
        canny_masekd_img_dir = parent_dir / 'canny_masked_img'

        path_list = [seghead_masekd_img_dir, sobel_masekd_img_dir, canny_masekd_img_dir]
        for path_dir in path_list:
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, exist_ok=True)

        iou_list = {"seghead":[],"sobel":[],"canny":[] }
        f1_list = {"seghead":[],"sobel":[],"canny":[] }

        max_diff_iou_sobel = -1
        max_diff_iou_canny = -1

        canny_win_count = 0 
        sobel_win_count = 0 

        sobel_over_sum = 0
        sobel_under_sum = 0

        canny_over_sum = 0
        canny_under_sum = 0

        for imgs, gts, names in tqdm(loader, desc="viz"):

            # ---- imgs to GPU (배치 유지) ----
            if not torch.is_tensor(imgs):
                imgs = torch.as_tensor(imgs)
            imgs = _ensure_3ch_float_tensor(imgs)          # [B,3,H,W]
            imgs = imgs.to(device, non_blocking=True)

            # B = imgs.shape[0]
            ori_size = (imgs.shape[-2], imgs.shape[-1])    # (H,W)

            # 공통 SAM encoding
            input_image_torch = preprocess(imgs).to(device)  # [B,3,1024,1024]
            image_embedding = sam_model.image_encoder(input_image_torch)

            # sobel ap & canny ap
            latent_sobel = ap_encoder_sobel(input_image_torch, image_embedding)
            latent_canny = ap_encoder_canny(input_image_torch, image_embedding)

            # each decoding processing
            pred_seghead = seg_decoder(image_embedding)
            pred_sobel = seg_decoder_sobel(latent_sobel)
            pred_canny = seg_decoder_canny(latent_canny)

            # postprocess -> 원본 해상도 [B,2,H,W]
            pred_seghead = postprocess_masks(pred_seghead, input_size=input_size, original_size=ori_size)
            pred_sobel = postprocess_masks(pred_sobel, input_size=input_size, original_size=ori_size)
            pred_canny = postprocess_masks(pred_canny, input_size=input_size, original_size=ori_size)

            gt_tensor = _gt_to_BHW(gts, device=pred_seghead.device)  # [B,H,W]

            # IoU 계산 (binary_iou가 target [B,1,H,W] 기대면 unsqueeze)
            pred_seghead_mask = torch.softmax(pred_seghead, dim=1).argmax(dim=1)  # [B,H,W]
            pred_sobel_mask = torch.softmax(pred_sobel, dim=1).argmax(dim=1)  # [B,H,W]
            pred_canny_mask = torch.softmax(pred_canny, dim=1).argmax(dim=1)  # [B,H,W]

            iou_seghead = binary_iou(pred_seghead_mask, gt_tensor.unsqueeze(1))
            iou_sobel = binary_iou(pred_sobel_mask, gt_tensor.unsqueeze(1))
            iou_canny = binary_iou(pred_canny_mask, gt_tensor.unsqueeze(1))

            iou_list['seghead'].extend(iou_seghead)
            iou_list['sobel'].extend(iou_sobel)
            iou_list['canny'].extend(iou_canny)

            # ---- F1-score 계산 추가 ----
            f1_list['seghead'].append(binary_f1(pred_seghead_mask, gt_tensor).item())
            f1_list['sobel'].append(binary_f1(pred_sobel_mask, gt_tensor).item())
            f1_list['canny'].append(binary_f1(pred_canny_mask, gt_tensor).item())

            for idx in range(len(iou_canny)):
                diff_canny = iou_canny[idx] - iou_seghead[idx]
                if diff_canny >= max_diff_iou_canny:
                    max_diff_iou_canny = diff_canny
                    max_diff_canny_img_name = names[idx]

                    seghead_masked_img_name = names[idx].replace(".png","_seg.png")
                    canny_masked_img_name = names[idx].replace(".png","_canny.png")

                    dst_seg_masked_path = canny_masekd_img_dir / seghead_masked_img_name
                    dst_canny_masked_path = canny_masekd_img_dir / canny_masked_img_name
                    origin_img = imgs[idx].cpu().numpy()

                    save_overlay_image(origin_img, pred_seghead[idx], save_path = dst_seg_masked_path)
                    save_overlay_image(origin_img, pred_canny[idx], save_path = dst_canny_masked_path)

                if diff_canny >= 0 :
                    canny_win_count += 1
                    canny_over_sum += diff_canny
                else:
                    canny_under_sum += diff_canny

            for idx in range(len(iou_sobel)):
                diff_sobel = iou_sobel[idx] - iou_seghead[idx]
                if diff_sobel >= max_diff_iou_sobel:
                    max_diff_iou_sobel = diff_sobel
                    max_diff_sobel_img_name = names[idx]

                    seghead_masked_img_name = names[idx].replace(".png","_seg.png")
                    sobel_masked_img_name = names[idx].replace(".png","_sobel.png")

                    dst_seg_masked_path = sobel_masekd_img_dir / seghead_masked_img_name
                    dst_sobel_masked_path = sobel_masekd_img_dir / sobel_masked_img_name
                    origin_img = imgs[idx].cpu().numpy()

                    save_overlay_image(origin_img, pred_seghead[idx], save_path = dst_seg_masked_path)
                    save_overlay_image(origin_img, pred_canny[idx], save_path = dst_sobel_masked_path)

                if diff_sobel >= 0 :
                    sobel_win_count += 1
                    sobel_over_sum += diff_sobel
                else:
                    sobel_under_sum += diff_sobel

        mean_seghead_iou =float(np.mean(iou_list['seghead'])) if iou_list['seghead'] else 0.0
        mean_sobel_iou =float(np.mean(iou_list['sobel'])) if iou_list['sobel'] else 0.0
        mean_canny_iou =float(np.mean(iou_list['canny'])) if iou_list['canny'] else 0.0

        mean_seghead_f1 = sum(f1_list['seghead']) / len(f1_list['seghead'])
        mean_sobel_f1 = sum(f1_list['sobel']) / len(f1_list['sobel'])
        mean_canny_f1 = sum(f1_list['canny']) / len(f1_list['canny'])

        dataset_count = len(loader.dataset) 
        sobel_win_rate = sobel_win_count / dataset_count
        canny_win_rate = sobel_win_count / dataset_count

        sobel_over_mean = sobel_over_sum / dataset_count
        canny_over_mean = canny_over_sum / dataset_count
        sobel_under_mean = sobel_under_sum / dataset_count
        canny_under_mean = canny_under_sum / dataset_count

        print(f"Model Type Name: {model_type}")
        print(f"IOU result || seghead:{mean_seghead_iou:.3f}, sobel:{mean_sobel_iou:.3f}, canny:{mean_canny_iou:.3f}")
        print(f"F1-score result || seghead:{mean_seghead_f1:.3f}, sobel:{mean_sobel_f1:.3f}, canny:{mean_canny_f1:.3f}")
        print(f"Maximun IOU diff: sobel-seghead:{max_diff_iou_sobel:.3f}, canny-seghead:{max_diff_iou_canny:.3f}")
        print(f"Maximun IOU diff image name: sobel-seghead:{max_diff_sobel_img_name}, canny-seghead:{max_diff_canny_img_name}")
        print(f"Better Performance than Seghead: sobel win rate:{sobel_win_rate:.3f}, canny win rate:{canny_win_rate:.3f}")
        print(f"Mean IOU over than Seghead: sobel:{sobel_over_mean:.3f}, canny :{canny_over_mean:.3f}")
        print(f"Mean IOU less than Seghead: sobel:{sobel_under_mean:.3f}, canny :{canny_under_mean:.3f}")
        print("-----------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
