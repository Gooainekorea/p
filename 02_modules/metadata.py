import pandas as pd
import json
from config_manager import ConfigManager

def metadata(convert_to_csv):
    config = ConfigManager()
    paths = config.path
    base_input_path = paths.get('base_input_path')
    output_path = paths.get('output_path')
    
    if convert_to_csv:
        metadata_path = paths.get('metadata_path')
        species_names_path = paths.get('species_names_path')

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
    else:
        # 아니면 불러오기
        metadata = pd.read_csv(f'{base_input_path}plantnet-metadata/metadata.csv')
        species_names = pd.read_csv(f'{base_input_path}plantnet-metadata/species_names.csv')


if __name__ == "__main__":
    metadata(True)
