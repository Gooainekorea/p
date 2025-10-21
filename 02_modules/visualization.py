# 데이터 처리 및 관리
import pandas as pd
import json

# 이미지 및 시각화
from PIL import Image
import matplotlib.pyplot as plt


# 프로젝트 내부 모듈
from config_manager import ConfigManager


# 경로
config = ConfigManager()
paths = config.path
output_path = paths.get('output_path')
input_path = paths.get('input_path')
base_input_path = paths.get('base_input_path')


# 이미지 보여주는 함수

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