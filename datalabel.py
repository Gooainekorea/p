import pandas as pd
import os

# 메타데이터와 클래스 정보 불러오기
metadata = pd.read_json('D:/ain2/PlantNet/output_data/matadata/metadata.json')
species_names = pd.read_json('D:/ain2/PlantNet/output_data/matadata/species_names.json')

# species_id와 species_name 매핑
species_id_to_name = dict(zip(species_names['species_id'], species_names['species_name']))

# 이미지별 라벨 매핑
image_label_map = {}
for _, row in metadata.iterrows():
    image_id = row['id']
    species_id = row['species_id']
    image_file = f"{image_id}.jpg"
    image_label_map[image_file] = species_id

# 예시: 특정 이미지의 라벨과 학명 확인
sample_image = '6033c318d5678da896eae5ae54ac60f71e5286bf.jpg'
species_id = image_label_map[sample_image]
species_name = species_id_to_name[species_id]
print(f"이미지: {sample_image}, 라벨: {species_id}, 학명: {species_name}")
