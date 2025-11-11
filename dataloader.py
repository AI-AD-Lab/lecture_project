import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import cv2

class ORFDDataset(Dataset):
    def __init__(self, dataset_root, mode='training', transform=None):
        """
        dataset_root: ORFD_dataset 상위 폴더 경로
        mode: 'training' 또는 'validation'
        transform: 데이터 변환 (torchvision.transforms 등)
        """
        if mode not in ['training', 'testing','validation']:
            raise ValueError("mode must be 'training' or 'testing' or 'validation' ")
        
        self.transform = transform
        self.mode = mode
        
        # 경로 설정
        base_dir = os.path.join(dataset_root, mode)
        self.image_dir = os.path.join(base_dir, 'image_data')
        self.gt_dir = os.path.join(base_dir, 'gt_image')
        
        # 이미지 파일 목록
        self.image_files = sorted(glob(os.path.join(self.image_dir, '*.png')))
        self.gt_files = sorted(glob(os.path.join(self.gt_dir, '*.png')))
        
        # 파일 이름 매칭
        self.pairs = []
        gt_map = {Path(p).stem.replace("_fillcolor", ""): p for p in self.gt_files}

        for img_path in self.image_files:
            stem = Path(img_path).stem
            if stem in gt_map:
                self.pairs.append((img_path, gt_map[stem]))
            else:
                print(f"[WARN] GT not found for {stem}.png")
    

        print(f"[{mode}] Loaded {len(self.pairs)} image pairs.")
    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        # cv2 로드 (BGR → RGB)
        image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        return rgb_image, gt, os.path.basename(img_path)

# 사용 예시
if __name__ == "__main__":
    from torchvision import transforms
    
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset_root = './ORFD_dataset'
    train_dataset = ORFDDataset(dataset_root, mode='training', transform=transform)
    val_dataset = ORFDDataset(dataset_root, mode='validation', transform=transform)
    
    print(len(train_dataset), "train samples")
    print(len(val_dataset), "validation samples")

    # 예시로 한 쌍 출력
    img, gt, _ = train_dataset[0]
    print("Image shape:", img.shape, "GT shape:", gt.shape)
