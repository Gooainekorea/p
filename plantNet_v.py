# 파일 열기
import PIL.Image
import PIL.ImageTk
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import torch
import torchvision.transforms as transforms

# config
base_input_path = 'D:/ain/PlantNet/' # 기본 입력경로 D:\ain\PlantNet
input_path = f'{base_input_path}plantnet_300K/' # 이미지 "D:\ain\PlantNet\plantnet_300K"
output_path = f'{base_input_path}output_data/' # 출력결과 "D:\ain\PlantNet\output_data"
metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일
species_names_path = f'{input_path}plantnet300K_species_id_2_name.json'#종이름파일
model_save_dir = f'D:\ain\PlantNet\output_data\models'#모델저장 경로


root = tk.Tk()
root.geometry("200x100")
lbl = tk.Label(text="LABEL")
btn = tk.Button(text="PUSH")
btn2 = tk.Button(text="버튼")

lbl.pack()
btn.pack()
btn2.pack()

def imageToData(filename): # 선택한 이미지 보이기, 이미지 전처리
    openImage = PIL.Image.open(filename)

    dispImage = PIL.ImageTk.PhotoImage(openImage.resize((300,300)))
    imageLabel.configure(image=dispImage)
    imageLabel.image=dispImage

    transform = transforms.Compose([
    transforms.Resize(256), # 이미지의 짧은 변 크기를 256으로 맞춤
    transforms.ToTensor(), # 텐서 변환 및 픽셀 값 정규화
    transforms.Normalize( # 채널별 정규화
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
    transforms.CenterCrop(224) #중앙에서 224x224 자름
    ])
    tensorImage = transform(openImage)
    return tensorImage


def predictDigits(data): # 학습된 모델을 로드.
    
    textLabel.configure(text="이 그림은"+str(n)+"입니다.")

def openFile():
    fpath=fd.askopenfilename()
    if fpath:
        data=imageToData(fpath)
        predictDigits((data))

root=tk.Tk()
root.geometry("400x400")

btn=tk.Button(root,text="파일열기", command=openFile)
imageLabel=tk.Label()

btn.pack()
imageLabel.pack()

textLabel=tk.Label(text="사진 인식중.")
textLabel.pack()

tk.mainloop()