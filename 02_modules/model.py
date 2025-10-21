# 모델구조 정의
import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
from config import num_classes


class PlantNet(nn.Module):
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss >= self.best_valid_loss:
            return

        self.best_valid_loss = current_valid_loss
        print(f"\nBest validation loss: {self.best_valid_loss}")
        print(f"\nSaving best model for epoch: {epoch+1}\n")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, best_model_path)
        print(f"모델 저장 성공. 경로 : {best_model_path}")
 