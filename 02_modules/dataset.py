import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import json



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
metadata.head() # 헤드확인
species_names.info()

metadata['species_id'] = pd.to_numeric(metadata['species_id'])
