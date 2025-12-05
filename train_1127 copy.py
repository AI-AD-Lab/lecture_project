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

####### Adaptive Patch 적용
from rod_effs_ap_model import RODSegAdaptivePatch

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

#####
"""
def build_optimizer(model, lr=1e-3):
    return AdamW(model.parameters(), lr=lr)
"""
###

def build_optimizer(model, lr=1e-4, weight_decay=1e-3):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)




# 3) Poly LR scheduler: lr = base_lr * (1 - iter/max_iter) ** power
def build_poly_scheduler(optimizer, total_steps, power=0.9):
    def poly_decay(step):
        step = min(step, total_steps)
        return (1 - step / float(total_steps)) ** power
    return LambdaLR(optimizer, lr_lambda=poly_decay)

###
"""
def preprocess(x):
    """"""Normalize pixel values and pad to a square input.""""""
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
    """"""
    logits: [B,2,H,W]  (배경/전경)
    target: [B,H,W]    (0/1, Long)
    """"""
    return F.cross_entropy(logits, target)
"""
###

@torch.no_grad()
def binary_iou(pred_mask: torch.Tensor, target : torch.Tensor) -> float:
    """
    pred_mask: [B,2,H,W] (0/1)
    target   : [B,1,H,W] (0/1)
    """

    # bool 로 변환

    pred = pred_mask.bool()
    tgt = target.bool()

    inter = (pred & tgt).sum(dim=(1,2)).float()
    union = (pred | tgt).sum(dim=(1,2)).float().clamp_min(1.0)
    
    return (inter / union).mean().item()

###
"""    
def binary_iou(pred_mask, target):
    """"""
    pred_mask: [B,2,H,W] (0/1)batch_size
    target   : [B,1,H,W] (0/1)
    """"""
    inter = (pred_mask & target).sum(dim=(1,2)).float()
    union = (pred_mask | target).sum(dim=(1,2)).float().clamp_min(1.0)
    return (inter / union).mean().item()
"""
###

###

def train_orfd_ap(
    dataset_root: str,
    model: nn.Module,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-4,
    #
    batch_size: int = 4,
    num_workers: int = 4,
    val_every: int = 1,
    model_tag: str = "effsam_vits_ap",
):
    """
    - dataset_root: ORFDDataset 루트 경로
    - model: RODSegAdaptivePatch 인스턴스
    
    """



    # Dataset / Loader
    train_ds = ORFDDataset(dataset_root, mode='training')
    # test_ds = ORFDDataset(dataset_root, mode='testing')
    val_ds   = ORFDDataset(dataset_root, mode='validation')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    #test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True,
                              #num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    dataset_loader = {'train':train_loader,
                      #'test':test_loader,
                      'validation':val_loader
                    }
    """
    # 모델(이미 존재한다고 가정)
    # efficientsam, seg_decoder, preprocess, transform 은 사용자 코드에서 그대로 사용
    sam_model.to(device).eval()
    seg_decoder.to(device).train()
    """
    model.to(device)
    model.train()

    # Optim / Scheduler
    # params = list(efficientsam.parameters()) + list(seg_decoder.parameters())
    #seg_decoder만 학습

    optimizer = build_optimizer(model, lr=lr, weight_decay=1e-3)
    total_steps = len(train_loader) * epochs
    scheduler = build_poly_scheduler(optimizer, total_steps, power=0.9)
    
    best_val_iou = 0.0
    
    # transform = ResizeLongestSide(1024)
    # print(f'Train {model_type_name}')

    print(f"Start training Adaptive Patch model: {model_tag}")
    global_step = 0

    for epoch in range(1, epochs+1):
        print(f'processing: {epoch}/{epochs+1}')
        #seg_decoder.train()
        t0 = time.time()
        # ========train=========
        model.train()
        train_running_loss = 0.0
        for imgs, gts, _ in dataset_loader['train']:
            # DataLoader가 batch_size=1일 때 rgb_image[0] 꺼내기
            imgs = imgs.to(device, dtype=torch.float32)

            # 채널 위치가 맨 뒤(NHWC)면 NCHW로 변환
            if imgs.ndim == 4 and imgs.shape[1] not in (1, 3):
                # 예: [B,H,W,3] -> [B,3,H,W]
                imgs = imgs.permute(0, 3, 1, 2).contiguous()

            gts = gts.to(device)
            if gts.ndim == 4 and gts.shape[1] == 1:
                gts = gts[:, 0, :, :]
            gts = gts.long()
            if gts.max() > 1:
                gts = (gts > 127).long()

            logits = model(imgs)            # [B,2,H,W]
            loss = F.cross_entropy(logits, gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_running_loss += loss.item() * imgs.size(0)
            """
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
            input_image_torch = preprocess(input_image_torch).to(demodel=vice)

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image_torch)

            '''
            vits -> embeddings => 
            [ [1, 256, 64, 64],  -> [0]은 1개 존재
              [[1, 64, 64, 384], -> [1][0]
              [1, 64, 64, 384]],  -> [1][1]
              , ....               -> [1]에 12개 존재model=
            ]
            '''

            # decoder는 학습 대상 (grad 계산 O)
            pred_mask = seg_decoder(image_embedding) # [1, 2, 256, 256] 반환
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

            if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 model=크기와 gt_image 크기 비교
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )

            # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280)
            gt = torch.as_tensor(gt, dtype=torch.long, device=pred_mask.device)
            if gt.ndim == 2:
                gt = gt.unsqueeze(0)  # [1, H, W] 형태로 맞춤model=

            # 🔥 0~255 → 0~1로 변환
            if gt.max() > 1:
                gt = (gt > 127).long()

            
            loss = torch.nn.functional.cross_entropy(pred_mask, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * gt.size(0)model=
            """

        train_loss = train_running_loss / len(train_loader.dataset)
        dt = time.time() - t0
        print(f"  Train loss: {train_loss:.4f}  ({dt:.1f}s)")
        # train_loss = running_loss / len(train_loader.dataset)
        # dt = time.time() - t0

        # ====== Validation ======

        if (epoch % val_every) == 0: 
            #print(f'Test {model_type_name}')
            #seg_decoder.eval()
            model.eval()
            val_running_loss = 0.0
            val_iou_list = []

            with torch.no_grad():
                for imgs, gts, _ in dataset_loader['validation']:
                    
                    imgs = imgs.to(device, dtype=torch.float32)

                    if imgs.ndim == 4 and imgs.shape[1] not in (1, 3):
                        imgs = imgs.permute(0, 3, 1, 2).contiguous()

                    gts = gts.to(device)
                    if gts.ndim == 4 and gts.shape[1] == 1:
                        gts = gts[:, 0, :, :]
                    gts = gts.long()
                    if gts.max() > 1:
                        gts = (gts > 127).long()

                    logits = model(imgs)
                    loss = F.cross_entropy(logits, gts)

                    val_running_loss += loss.item() * imgs.size(0)

                    # mIoU 계산
                    preds = torch.softmax(logits, dim=1).argmax(dim=1)  # [B,H,W] 0/1
                    val_iou_list.append(binary_iou(preds, gts))
                    """
                    rgb_image = imgs[0].numpy() if isinstance(imgs, torch.Tensor) else imgs
                    gt = gts[0].numpy() if isinstance(gts, torch.Tensor) else gts

                    # 원본 크기
                    ori_size = rgb_image.shape[:2] # (720, 1280)
                    # transform 적용
                    input_image = transform.apply_image(rgb_image)model=
                    input_size = input_image.shape[:2] # (720,1280) -> (576,1024)로 변환 ResizeLongestSide(1024)

                    # torch 텐서 변환 [1, 3, 576, 1024] -> [1, 3, 1024, 1024]
                    input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()[None, :, :, :]
                    input_image_torch = preprocess(input_image_torch).to(device)

                    image_embedding = sam_model.image_encoder(input_image_torch)

                    # decoder는 학습 대상 (grad 계산 O)
                    pred_mask = seg_decoder(image_embedding) # [1, model=2, 256, 256] 반환
                    pred_mask = postprocess_masks(pred_mask, input_size, ori_size) # [1, 2, 720, 1280] 반환 -> 원본크기의 2채널

                    if pred_mask.shape[-2:] != gt.shape[-2:]: # 예측 마스크 크기와 gt_image 크기 비교
                        pred_mask = torch.nn.functional.interpolate(
                            pred_mask, size=gt.shape[-2:], mode='bilinear', align_corners=False
                        )

                    # ✅ 타깃 텐서 변환 (720, 1280) -> (1, 720, 1280model=)
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
                    """
            val_loss = val_running_loss  / len(val_loader.dataset)
            mean_iou = np.mean(val_iou_list) if val_iou_list else 0.0
            print(f"  Val  loss: {val_loss:.4f}  mIoU: {mean_iou:.4f}")
            #print(f"[Epoch {epoch:03d}] "
            #      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mIoU={mean_iou:.4f}  ({dt:.1f}s)")

            # 체크포인트 저장
            if mean_iou > best_val_iou:
                best_val_iou = mean_iou
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_iou": best_val_iou,
                }
                os.makedirs("ckpts", exist_ok=True)
                save_path = f"ckpts/best_{model_tag}.pth"
                torch.save(ckpt, save_path)
                print(f"  -> New best mIoU={best_val_iou:.4f}. checkpoint saved to {save_path}")

        
    print("Training done.")
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

"""
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
    for model_type in model_types:

        print(f'Now Loaddig.... {model_type}') # 모델 타입에 따른 설정
        sam_model = sam_model_dict[model_type](checkpoint=sam_model_weight_chekpoint[model_type])
        seg_decoder = SegHead(sam_variant=model_type)

        train_orfd(
            dataset_root = image_file,
            sam_model=sam_model,
            seg_decoder=seg_decoder,
            device = device,
            epochs= 10,
            batch_size=1,
            num_workers=1,
            val_every=2,
            model_type_name = model_type
        )
"""

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_root = "./ORFD_dataset"

    sam_ckpt = "./weights/efficient_sam_vits.pt"

    model = RODSegAdaptivePatch(
        sam_ckpt=sam_ckpt,
        num_classes=2,
        im_size=1024,
        boundary_thresh=0.0,
    )

    train_orfd_ap(
        dataset_root=dataset_root,
        model=model,
        device=device,
        epochs=10,
        lr=1e-4,
        batch_size=4,
        num_workers=4,
        val_every=1,
        model_tag="rod_effs_ap_vits",
    )


if __name__ == '__main__':
    main()


"""
 정리하자면
 1.sam_model / seg_decoder 제거함
    sam_model = sam_model_dict[model_type](...)
    seg_decoder = SegHead(...)
    image_embedding = sam_model.image_encoder(...)
    pred_mask = seg_decoder(image_embedding)

    대신 한 줄:
    logits = model(imgs) 

2.Adaptive Patch는 전부 모델 안에서 처리

    EffSamViTSBackboneAP + BoundaryScoreModule + local_refine + decoder

    train loop는 “그냥 segmentation 모델 학습”만 신경 쓰면 됨.

3. IoU/손실 계산 정리

    cross_entropy(logits, gts)

    preds = softmax(logits).argmax(dim=1) → [B,H,W]

    0/255 라벨은 (gt > 127).long() 으로 0/1로 변환

    binary_iou(preds, gts)는 [B,H,W] 기준으로 계산

4. checkpoint

    model.state_dict()만 저장 (sam_model, seg_decoder 구분 없음)
"""