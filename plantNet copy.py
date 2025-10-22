import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm #실시간 진행 막대그래프

# config
base_input_path = 'D:/ain/PlantNet/' # 기본 입력경로 D:\ain\PlantNet
input_path = f'{base_input_path}plantnet_300K/' # 이미지 "D:\ain\PlantNet\plantnet_300K"
output_path = f'{base_input_path}output_data/' # 출력결과 "D:\ain\PlantNet\output_data"
metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일
species_names_path = f'{input_path}class_idx_to_species_id.json'#클래스 인덱스(0~1080)와 종 ID를 매핑
species_id_path = f'{input_path}plantnet300K_species_id_2_name.json'#종 ID와 학명을 매핑
convert_to_csv = True #csv변환여부

# --- 수정된 부분 시작 ---
# 출력 및 모델 저장 경로가 없으면 자동으로 생성
model_save_dir = os.path.join(output_path, 'models')
os.makedirs(output_path, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
print(f"출력 폴더가 생성되었습니다: {output_path}")
print(f"모델 저장 폴더가 생성되었습니다: {model_save_dir}")
# --- 수정된 부분 끝 ---

if convert_to_csv:
    metadata = pd.read_json(metadata_path) 
    metadata = metadata.transpose() # w전치
    metadata = metadata.reset_index() # 열이름을 id로 변환
    metadata = metadata.rename(columns={"index": "id"})
    metadata.to_csv(f'{output_path}metadata.csv', index=False)

    species_names = pd.read_json(species_names_path, orient="index")
    species_names = species_names.reset_index()
    species_names = species_names.rename(columns={"index": "species_id", 0: "species_name"})
    species_names.to_csv(f'{output_path}species_names.csv', index=False)
else:
   # load from pre-converted dataset
   metadata = pd.read_csv(f'{base_input_path}plantnet-metadata/metadata.csv')
   species_names = pd.read_csv(f'{base_input_path}plantnet-metadata/species_names.csv')

def show_image(path):
    img = Image.open(f'{input_path}{path}')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_tensor_image(tensor):
    tensor = tensor.detach().numpy().transpose((1, 2, 0))
    plt.imshow(image_array)
    plt.axis('off')  # To hide axis values
    plt.show()

metadata.info() # 각 열에 대한 정보

# json 파일로 데이터 프레임 만들어지는거 확인함 --------------------------------------------------------------
# 데이터 불균형 확인
metadata.head() # 헤드확인
species_names.info()

metadata['species_id'] = pd.to_numeric(metadata['species_id'])

#--------------------------------------------------------------
#헬퍼
plt.style.use('ggplot')

# 모델 저장 경로를 위에서 생성한 폴더 기준으로 설정
best_model_path = os.path.join(model_save_dir, 'best_model.pth')

class SaveBestModel:
    """
    훈련 중에 최고의 모델을 저장하는 클래스. 
    검증 손실이 이전 최소값보다 작으면 저장합니다
    """
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
        

def load_best_model():
    return torch.load(best_model_path)

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

def show_species_sample(species_id):
    directory_path = f"{input_path}images/train/{species_id}/"
    all_files = os.listdir(directory_path)
    image_files = [f for f in all_files if f.lower().endswith(('jpg'))]
    random_image_file = random.choice(image_files)
    image_path = os.path.join(directory_path, random_image_file)
    image = Image.open(image_path)
    return image
#--------------------------------------------------------------
# 모델과 데이터 불러옴

#이미지 변환 처리
train_transform = transforms.Compose([ #여러 변환을 순차적으로 연결
    transforms.Resize((256, 256)), # 가로 세로 256픽셀
    transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 이미지를 좌우 반전
    transforms.RandomVerticalFlip(p=0.5), # 50% 확률로 이미지를 좌우 반전
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), #가우시안 블러를 적용하여 노이즈를 추가
    transforms.RandomRotation(45), #-45도에서 +45도 범위 내에서 무작위로 회전
    transforms.ToTensor(), #이미지를 PyTorch 텐서로 변환하고, 픽셀값을 범위로 정규화
    transforms.Normalize( # 채널별 정규화 평균 0.5, 표준편차 0.5를 사용
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5]
    ),
    transforms.RandomCrop(224) # 224x224 무작위 잘라내기
])


valid_transform = transforms.Compose([
    transforms.Resize(256), # 이미지의 짧은 변 크기를 256으로 맞춤
    transforms.ToTensor(), # 텐서 변환 및 픽셀 값 정규화
    transforms.Normalize( # 채널별 정규화
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
    transforms.CenterCrop(224) #중앙에서 224x224 자름
])

train_dataset = datasets.ImageFolder( # 폴더 구조를 기준으로 이미지와 라벨을 자동으로 매핑해 데이터셋 생성
    root=f'{input_path}images/train/', # 학습 이미지 경로
    transform=train_transform # 앞서 만든 파이프라인적용
)

test_dataset = datasets.ImageFolder( #폴더로부터 데이터를 읽어 valid_transform 적용
    root=f'{input_path}images/test/',
    transform=valid_transform
)

batch_size = 32
subset = None
train_subset = None
test_subset = None

if subset is not None: # 일부 데이터만 사용하고 싶을때 subset 값이 있다면
    print(f"subsetting data to {subset} results") # 몇 개의 데이터만 사용할지 출력 확인

    # subset이 전체 데이터 개수보다 작으면 subset만큼, 아니면 전체 데이터 개수만큼 인덱스 생성
    train_subset_indices = list(range(subset if subset < len(train_dataset) else len(train_dataset)))
    # train_dataset에서 위에서 만든 인덱스만큼만 골라서 부분 데이터셋 생성    
    train_subset = Subset(train_dataset, train_subset_indices)

    # test_dataset도 train_dataset과 같은 방식으로 부분집합 인덱스 생성
    test_subset_indices = list(range(subset if subset < len(test_dataset) else len(test_dataset)))
    # test_dataset의 부분집합 생성
    test_subset = Subset(test_dataset, test_subset_indices)

# DataLoader 생성 - train_subset이 None이면 전체 train_dataset을 사용하고, 아니면 부분집합을 사용
train_loader = DataLoader( 
    train_dataset if train_subset is None else train_subset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=False
)

test_loader = DataLoader(
    test_dataset if test_subset is None else test_subset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=False
)

# 모델 정의 (중복되는 부분 정리)
model = alexnet(weights=AlexNet_Weights.DEFAULT)

num_classes = len(species_names)
# 마지막 레이어만 교체
model.classifier[-1] = torch.nn.Linear(4096, num_classes)

# 마지막 레이어를 제외한 모든 파라미터를 동결
for param in model.parameters():
    param.requires_grad = False

# 마지막 레이어의 파라미터만 학습하도록 설정
for param in model.classifier[6].parameters():
    param.requires_grad = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)

#----------------------------------------
# 훈련 및 검증 로직
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# SaveBestModel 인스턴스 생성
save_best_model = SaveBestModel()

print("\n스크립트 초기 설정이 완료되었습니다. 훈련을 시작할 준비가 되었습니다.")



#---------------------


def train():
    epochs =20

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        
        # --- 모델 훈련(Training) ---
        model.train()  # 모델을 훈련 모드로 설정
        train_running_loss = 0.0
        # tqdm을 사용해 진행률 표시
        prog_bar = tqdm(train_loader, desc="Training", leave=False)
        for i, data in enumerate(prog_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
            
            outputs = model(images)  # 모델의 예측값 계산
            loss = criterion(outputs, labels)  # 손실 계산
            
            loss.backward()  # 역전파를 통해 기울기 계산
            optimizer.step()  # 옵티마이저를 통해 가중치 업데이트
            
            train_running_loss += loss.item()
            
        train_loss = train_running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # --- 모델 검증(Validation) ---
        model.eval()  # 모델을 평가 모드로 설정
        valid_running_loss = 0.0
        with torch.no_grad():  # 기울기 계산 비활성화
            prog_bar = tqdm(test_loader, desc="Validating", leave=False)
            for i, data in enumerate(prog_bar):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                valid_running_loss += loss.item()
                
        valid_loss = valid_running_loss / len(test_loader)
        valid_losses.append(valid_loss)
        
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        
        # 현재 에폭의 검증 손실을 기준으로 최고의 모델을 저장
        save_best_model(valid_loss, epoch, model, optimizer, criterion)

    print('훈련이 완료되었습니다.')

    # 훈련 과정의 손실 그래프 그리기
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color='green', linestyle='-', label='train loss')
    plt.plot(valid_losses, color='blue', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_path}models/loss.png')
    plt.show()

    print(f"손실 그래프가 {output_path}models/loss.png 에 저장되었습니다.")

#=================================================================================
# 모델 저장

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    train()