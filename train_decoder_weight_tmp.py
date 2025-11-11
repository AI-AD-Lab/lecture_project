import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import numpy as np

# --- 1. 모델 및 데이터셋 정의 파일 임포트 (가정) ---
# 실제 파일 경로에 맞게 수정 필요
from seg_decoder import SegHead
from efficient_sam.efficient_sam_encoder import ImageEncoderViT 
from datasets.RELLIS_3D_dataset import RELLIS3DDataset 
# from your_project.utils import DiceLoss, iou_metric 

# *주의: 실제 환경에서는 위의 주석을 해제하고 경로를 맞추세요.*

# 임시 정의 (실제 코드로 대체 필요)
class SegHead(nn.Module):
    def __init__(self):
        super().__init__()
        # SAMAggregatorNeck과 SegHead를 통합한 간소화된 구조
        self.conv = nn.Conv2d(1280, 2, kernel_size=1) 
    def forward(self, inputs):
        # ImageEncoderViT의 튜플 출력을 받음: (final_embedding, inner_states)
        final_embedding, inner_states = inputs
        # SegHead의 최종 출력을 256x256 로짓으로 가정
        return F.interpolate(self.conv(final_embedding), size=(256, 256), mode='bilinear', align_corners=False)

class ImageEncoderViT(nn.Module):
    def __init__(self):
        super().__init__()
        # SAM 인코더 역할 (가중치는 로드되었다고 가정)
        self.dummy_output = nn.Parameter(torch.randn(1, 1280, 64, 64)) 
        self.dummy_states = [torch.randn(1, 64, 64, 384)] * 12
    def forward(self, x):
        # 실제 인코더는 로드된 가중치를 사용하며, 튜플을 출력합니다.
        # 여기서는 설명을 위해 더미(Dummy) 데이터를 사용합니다.
        return self.dummy_output.repeat(x.size(0), 1, 1, 1), self.dummy_states 

# ************************************************

# --- 2. 손실 함수 및 평가 지표 (예시) ---
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        # 일반적으로 Dice Loss가 경계 학습에 유리합니다.
        # self.dice_loss = DiceLoss() 

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        # dice = self.dice_loss(F.softmax(pred, dim=1)[:, 1], target.float())
        return ce # + dice

# --- 3. 훈련 및 검증 함수 ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        images = data['rgb_image'].to(device)
        # 레이블은 Long 타입이어야 nn.CrossEntropyLoss에 적합합니다.
        labels = data['label'].long().to(device) 

        # 인코더를 no_grad로 감싸 가중치 업데이트 방지 (가장 중요)
        with torch.no_grad():
            image_embedding, inner_states = model['encoder'](images)
            
        inputs_for_decoder = (image_embedding, inner_states)
        pred_logits = model['decoder'](inputs_for_decoder)
        
        loss = criterion(pred_logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_model(model, dataloader, device):
    model['decoder'].eval()
    total_iou = 0
    with torch.no_grad():
        for data in dataloader:
            images = data['rgb_image'].to(device)
            labels = data['label'].long().to(device)

            image_embedding, inner_states = model['encoder'](images)
            inputs_for_decoder = (image_embedding, inner_states)
            pred_logits = model['decoder'](inputs_for_decoder)
            
            # 예측 마스크 (Freepace 클래스)
            predicted_mask = torch.argmax(pred_logits, dim=1) 
            
            # IoU 계산 (실제 iou_metric 함수로 대체 필요)
            # iou = iou_metric(predicted_mask, labels) 
            total_iou += 1 # 임시 값
    return total_iou / len(dataloader)


# --- 4. 메인 실행 루프 (best_epoch.pth 생성 로직 포함) ---

def main_train_script():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4.1 모델 로드 및 동결 설정
    image_encoder = ImageEncoderViT().to(device)
    # image_encoder.load_state_dict(torch.load('weights/sam_vit_h_4b8939.pth')['model'], strict=False)
    
    # 인코더 파라미터 동결 (필수!)
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    seg_decoder = SegHead().to(device) # 디코더는 랜덤 초기화 상태
    
    model = {'encoder': image_encoder, 'decoder': seg_decoder}

    # 4.2 데이터 로더 (RELLIS3DDataset 및 DataLoader 사용)
    # train_dataset = RELLIS3DDataset(root='your_path', mode='train')
    # val_dataset = RELLIS3DDataset(root='your_path', mode='val')
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # *주의: 실제 환경에서는 위의 주석을 해제하고 DataLoader를 설정해야 합니다.*
    
    # 4.3 최적화 설정
    criterion = CombinedLoss()
    # 옵티마이저는 디코더 파라미터만 학습하도록 지정 (필수!)
    optimizer = optim.AdamW(seg_decoder.parameters(), lr=1e-4) 

    # 4.4 훈련 루프 및 저장
    num_epochs = 50
    best_iou = -1.0
    ckpt_dir = 'ckpts/orfd'
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # val_iou = validate_model(model, val_loader, device)

        # *주의: 실제 IoU와 Loss 로그 출력 코드가 필요합니다.*

        # 💥 best_epoch.pth 파일 생성 로직 (가장 중요)
        # if val_iou > best_iou: 
        #     best_iou = val_iou
        #     torch.save(seg_decoder.state_dict(), os.path.join(ckpt_dir, 'best_epoch.pth'))
        #     print(f"Epoch {epoch+1}: New best IoU {best_iou:.4f}. Saved best_epoch.pth")
        
        # 임시 출력
        print(f"Epoch {epoch+1} completed. best_epoch.pth file is created upon successful validation.")


if __name__ == '__main__':
    main_train_script()