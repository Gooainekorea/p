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
base_input_path = 'D:/ain2/PlantNet/' # 기본 입력경로 D:\ain\PlantNet
input_path = f'{base_input_path}plantnet_300K/' # 이미지 "D:\ain\PlantNet\plantnet_300K"
output_path = f'{base_input_path}output_data/' # 출력결과 "D:\ain\PlantNet\output_data"
plantnet_metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일
species_idx_path = f'{input_path}class_idx_to_species_id.json'#종 id 파일
species_name_path = f'{input_path}plantnet300K_species_id_2_name.json'#학명 파일
convert_to_csv = True #csv변환여부


metadata = pd.read_json(f'{output_path}metadata/metadata.json')
species_names = pd.read_json(f'{output_path}metadata/species_names.json')


def show_image(path):
    img = Image.open(f'{input_path}{path}')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def show_tensor_image(tensor):
    tensor = tensor.detach().numpy().transpose((1, 2, 0))
    plt.imshow(tensor)
    plt.axis('off')  # To hide axis values
    plt.show()
   
#--------------------------------------------------------------


plt.style.use('ggplot')

model_path = os.path.join(output_path, 'models')
os.makedirs(model_path, exist_ok=True)
best_model_path = os.path.join(model_path, 'best_model.pth')

class SaveBestModel:
    """
    훈련 중에 최고의 모델을 저장하는 클래스.
    검증 손실이 이전 최소값보다 작으면 저장합니다
    """
    def __init__(self, model_class, *model_args, **model_kwargs):
        self.best_valid_loss = float('inf')
        self.model_class = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model_args = 
        self.model_kwargs = 

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss >= self.best_valid_loss:
            return
        self.best_valid_loss = current_valid_loss
        print(f"\nBest validation loss: {self.best_valid_loss}")
        print(f"\nSaving best model for epoch: {epoch+1}\n")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_name': type(criterion).__name__,
            'loss_params': criterion.__dict__
        }, best_model_path)
        print(f"모델 저장 성공. 경로 : {best_model_path}")

    def load_best_model(self):
        # 모델 객체를 내부에서 생성
        model = self.model_class(*self.model_args, **self.model_kwargs)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def load_model_data(self, model):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_name = checkpoint['loss_name']
        loss_params = checkpoint['loss_params']
        return model, optimizer, epoch, loss_name, loss_params

    def save_model(self, epochs, model, optimizer, criterion):
        """
        Function to save the trained model to disk.
        """
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_name': type(criterion).__name__,
            'loss_params': criterion.__dict__
        }, f'{output_path}models/model_{epochs}.pth')






def show_species_sample(species_id):
    # List all files in the directory
    directory_path = f"{input_path}images/train/{species_id}/"
    all_files = os.listdir(directory_path)


    # Filter out any non-image files if needed (e.g., based on file extension)
    image_files = [f for f in all_files if f.lower().endswith(('jpg'))]


    # Select a random image file
    random_image_file = random.choice(image_files)


    # Open and display the image
    image_path = os.path.join(directory_path, random_image_file)
    image = Image.open(image_path)
#     plt.imshow(image)
#     plt.axis('off')  # Hide the axis values
#     plt.show()
    return image
#--------------------------------------------------------------
# 모델과 데이터 불러옴
import torchvision.datasets as datasets
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
    transforms.RandomCrop(224) #random on training, center on validation
])




valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
    transforms.CenterCrop(224)
])


train_dataset = datasets.ImageFolder(
    root=f'{input_path}images/train/',
    transform=train_transform
)


test_dataset = datasets.ImageFolder(
    root=f'{input_path}images/test/',
    transform=valid_transform
)


from torch.utils.data import DataLoader, Subset


batch_size = 32
subset = 100
train_subset = None
test_subset = None


if subset is not None:
    print(f"subsetting data to {subset} results")
    train_subset_indices = list(range(subset if subset < len(train_dataset) else len(train_dataset)))
    train_subset = Subset(train_dataset, train_subset_indices)


    test_subset_indices = list(range(subset if subset < len(test_dataset) else len(test_dataset)))
    test_subset = Subset(test_dataset, test_subset_indices)


# DataLoader
train_loader = DataLoader(
    train_subset if train_subset is not None else train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_subset if test_subset is not None else test_dataset,
    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
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
for param in model.classifier[-1].parameters():
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


