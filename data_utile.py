import pandas as pd
import os

# config
base_input_path = 'D:/ain2/PlantNet/' # 기본 입력경로 D:\ain\PlantNet
input_path = f'{base_input_path}plantnet_300K/' # 이미지 "D:\ain\PlantNet\plantnet_300K"
output_path = f'{base_input_path}output_data/' # 출력결과 "D:\ain\PlantNet\output_data"
plantnet_metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일
species_idx_path = f'{input_path}class_idx_to_species_id.json'#종 id 파일
species_name_path = f'{input_path}plantnet300K_species_id_2_name.json'#학명 파일
convert_to_csv = True #csv변환여부

# 출력 및 모델 저장 경로가 없으면 자동으로 생성
metadata_dir = os.path.join(output_path, 'metadata')
os.makedirs(output_path, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)
print(f"출력 폴더가 생성되었습니다: {output_path}")
print(f"메타데이터 저장 폴더가 생성되었습니다: {metadata_dir}")

if convert_to_csv:
    metadata = pd.read_json(plantnet_metadata_path)
    metadata = metadata.transpose() # w전치
    metadata = metadata.reset_index() # 열이름을 id로 변환
    metadata = metadata.rename(columns={"index": "id"})
    metadata.to_json(f'{output_path}metadata/metadata.json', index=False)

    species_idx = pd.read_json(species_idx_path, orient="index")
    species_idx = species_idx.reset_index()
    species_idx = species_idx.rename(columns={"index": "species_idx", 0: "species_id"})
    # species_idx.to_csv(f'{output_path}species_names.csv', index=False)

    species_names = pd.read_json(species_name_path, orient="index")
    species_names = species_names.reset_index()
    species_names = species_names.rename(columns={"index": "species_id", 0: "species_name"})
    species_names = pd.merge(species_idx, species_names, on='species_id', how='left')
    species_names.to_json(f'{output_path}metadata/species_names.json', index=False)
else:
   # load from pre-converted dataset
   metadata = pd.read_json(f'{output_path}metadata/metadata.json')
   species_names = pd.read_json(f'{output_path}metadata/species_names.json')


