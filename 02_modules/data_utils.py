# 데이터 처리 및 관리
import pandas as pd
import json

# 프로젝트 내부 모듈
from config_manager import ConfigManager

# 공통 부분 밖으로 뺌
config = ConfigManager()
paths = config.path
output_path = paths.get('output_path')
metadata_path = paths.get('metadata_path')
species_names_path = paths.get('species_names_path')

def metadata_set(): #json 메타데이터 csv파일로 만들기
    metadata = pd.read_json(metadata_path) 
    metadata = metadata.transpose() # 전치
    metadata = metadata.reset_index() # 열이름을 id로 변환
    metadata = metadata.rename(columns={"index": "id"})
    metadata.to_csv(f'{output_path}metadata.csv', index=False)

    species_names = pd.read_json(species_names_path, orient="index")
    species_names = species_names.reset_index()
    species_names = species_names.rename(columns={"index": "species_id", 0: "species_name"})
    species_names.to_csv(f'{output_path}species_names.csv', index=False)

    num_classes = num_classes = len(species_names)  # 또는 len(species_df)
    all_config = config.get_all_config()
    all_config['hyperparameter']['num_classes'] = num_classes

    # config.json 파일에 변경된 내용을 다시 쓰기
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(all_config, f, indent=2)
    print("'config.json' 파일의 num_classes 값이 업데이트 되었습니다.")
    print(f"Preprocessed data saved to {output_path}")


def metadata_load():
    # 불러오기
    metadata = pd.read_csv(f'{output_path}metadata.csv')
    species_names = pd.read_csv(f'{output_path}species_names.csv')
    # 형변환. 안해주면 검색이었나 데이터 처리시 에러났었음
    metadata['species_id'] = pd.to_numeric(metadata['species_id'])
    
    return metadata, species_names

# 테스트용
# if __name__ == "__main__":
#     metadata_set()
#     metadata, species_names = metadata_load()
#     print(metadata)
#     print(species_names)