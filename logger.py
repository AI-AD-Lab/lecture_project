import os
import csv
from datetime import datetime

class TrainLogger:
    def __init__(self, save_path: str):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 학습 시작 시 날짜 고정 (YYMMDD)
        self.date = datetime.now().strftime("%y%m%d")

        # 첫 실행이면 헤더 작성
        if not os.path.exists(save_path):
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Epoch", 
                    "Train Loss", "Train mIoU", 
                    "Test Loss", "Test mIoU",
                    "Val Loss", "Val mIoU"
                ])

    def log(self, epoch: int, train_loss: float, train_miou: float,
            test_loss: float, test_miou: float,
            val_loss: float, val_miou: float):

        with open(self.save_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                 epoch,
                round(train_loss, 6), round(train_miou, 6),
                round(test_loss, 6), round(test_miou, 6),
                round(val_loss, 6), round(val_miou, 6)
            ])


if __name__ == '__main__':
    epoch = 1
    train_loss = 0.007430506220363854
    train_miou = 0.9918639426917776
    test_loss = 0.508762608388438
    test_miou = 0.9169812886289848
    val_loss = 1.8193448571555586
    val_miou = 0.7027280970988982

    date = datetime.now().strftime("%y%m%d")
    log_path = f"ckpts_adaptive/_train_log_{date}.txt"

    logger = TrainLogger(log_path)
    logger.log(epoch, train_loss, train_miou,train_loss, train_miou, train_loss, train_miou )
    logger.log(epoch, train_loss, train_miou,train_loss, train_miou, train_loss, train_miou )
    logger.log(epoch, train_loss, train_miou,train_loss, train_miou, train_loss, train_miou )
    logger.log(epoch, train_loss, train_miou,train_loss, train_miou, train_loss, train_miou )
    logger.log(epoch, train_loss, train_miou,train_loss, train_miou, train_loss, train_miou )

