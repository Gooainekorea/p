# 데이터 처리 및 관리
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, Dataset
from config_manager import ConfigManager


# 경로
config = ConfigManager()
paths = config.path
output_path = paths.get('output_path')
input_path = paths.get('input_path')
base_input_path = paths.get('base_input_path')
model_save_dir = paths.get('model_save_dir')

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomCrop(224)
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.CenterCrop(224)
    ])
    return train_transform, valid_transform

def get_datasets(input_path, train_transform, valid_transform):
    train_dataset = datasets.ImageFolder(root=f"{input_path}images/train/", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=f"{input_path}images/test/", transform=valid_transform)
    return train_dataset, test_dataset

def get_dataloaders(train_dataset, test_dataset, batch_size, subset=None):
    if subset is not None:
        train_subset = Subset(train_dataset, list(range(min(subset, len(train_dataset)))))
        test_subset = Subset(test_dataset, list(range(min(subset, len(test_dataset)))))
    else:
        train_subset = train_dataset
        test_subset = test_dataset
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, test_loader

# ==============================================================
#이미지 변환 처리
train_transform = transforms.Compose([ #여러 변환을 순차적으로 연결(파이프라인)
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



class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    def __getitem__(self, idx):
        return self.dataset[idx]
    def __len__(self):
        return len(self.dataset)