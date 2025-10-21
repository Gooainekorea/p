# 데이터 처리 및 관리
import os
import json
import pandas as pd


# PyTorch 및 관련 모듈
import torch
from torchvision.models import alexnet, AlexNet_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 프로젝트 내부 모듈
import data_utils
import visualization
from config_manager import ConfigManager

# 경로
config = ConfigManager()
paths = config.path
output_path = paths.get('output_path')
input_path = paths.get('input_path')
base_input_path = paths.get('base_input_path')
model_save_dir = paths.get('model_save_dir')

# 모델 저장경로 정의
best_model_path = os.path.join(model_save_dir, 'best_model.pth')

# 모델 저장경로 추가
all_config = config.get_all_config()
all_config['path']['best_model_path'] = best_model_path

# 검증 손실이 이전 최소값보다 작으면 저장 - 최적 모델 저장
class SaveBestModel:
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
 

# 모든 에폭 모델을 저장 - 학습 진행 상황 기록
def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    # 일반 저장 시에도 위에서 생성한 폴더 경로를 사용
    model_epoch_path = os.path.join(model_save_dir, f'model_{epochs}.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, model_epoch_path)
    print(f"Epoch {epochs} 모델 저장 성공. 경로: {model_epoch_path}")

def load_best_model():
    return torch.load(best_model_path)

def show_species_sample(species_id):
    directory_path = f"{input_path}images/train/{species_id}/"
    all_files = os.listdir(directory_path)
    image_files = [f for f in all_files if f.lower().endswith(('jpg'))]
    random_image_file = random.choice(image_files)
    image_path = os.path.join(directory_path, random_image_file)
    image = Image.open(image_path)
    return image