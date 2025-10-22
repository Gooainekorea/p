# 파일 열기
import PIL.Image
import PIL.ImageTk
import tkinter as tk
import tkinter.filedialog as fd
import torch
import torchvision.transforms as transforms
import pandas as pd
import plantnet

# config
base_input_path = 'D:/ain2/PlantNet/' # 기본 입력경로 D:\ain\PlantNet
input_path = f'{base_input_path}plantnet_300K/' # 이미지 "D:\ain\PlantNet\plantnet_300K"
output_path = f'{base_input_path}output_data/' # 출력결과 "D:\ain\PlantNet\output_data"
plantnet_metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일
species_idx_path = f'{input_path}class_idx_to_species_id.json'#종 id 파일
species_name_path = f'{input_path}plantnet300K_species_id_2_name.json'#학명 파일
convert_to_csv = True #csv변환여부

model = plantnet.SaveBestModel().load_best_model()

metadata = pd.read_json(f'{output_path}metadata/metadata.json')
species_names = pd.read_csv(f'{output_path}metadata/species_names.csv')
species_names = dict(zip(species_names['species_id'], species_names['species_name']))


model.eval()

def imageToData(filename): # 선택한 이미지 보이기, 이미지 전처리
    openImage = PIL.Image.open(filename)

    dispImage = PIL.ImageTk.PhotoImage(openImage.resize((300,300)))
    imageLabel.configure(image = dispImage)
    imageLabel.image = dispImage

    transform = transforms.Compose([
    transforms.Resize(256), # 이미지의 짧은 변 크기를 256으로 맞춤
    transforms.ToTensor(), # 텐서 변환 및 픽셀 값 정규화
    transforms.Normalize( # 채널별 정규화
        mean=[0.5]*3,
        std=[0.5]*3
    ),
    transforms.CenterCrop(224) #중앙에서 224x224 자름
    ])
    tensorImage = transform(openImage).unsqueeze(0)
    return tensorImage


def predictDigits(data): # 학습된 모델을 로드.
    with torch.no_grad():
        output = model(data)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        name = species_names.get(class_idx, "Unknown")

        # 확률 계산
        s_max = torch.softmax(output, 1) # 각 클래스에 대해 샘플이 속할 확률
        max_prob = torch.max(s_max).item() # 가장 높은 확률 반환
        prob = str(round(max_prob * 100, 2)) # 백분율로 변환 및 반올림

    textLabel.configure(text="이 식물의 학명은"+str(name)+"일 확률이"+(prob)+"% 입니다.")

def openFile():
    fpath=fd.askopenfilename()
    if fpath:
        data=imageToData(fpath)
        predictDigits((data))



root = tk.Tk()
root.geometry("400x400")

btn = tk.Button(root, text="파일 열기", command=openFile)
btn.pack(pady=10)

imageLabel = tk.Label(root)
imageLabel.pack(pady=10)

textLabel = tk.Label(root, text="사진 인식 중.")
textLabel.pack(pady=10)
