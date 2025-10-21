import json
import os

class ConfigManager:
    _instance = None
    _initialized = False

    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_filename='config.json'):
        if self._initialized:
            return

        # 파일 못찾아서 절대 경로 찾아서 만들기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_filename)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"'{config_path}' 파일을 찾을 수 없습니다.")
        except json.JSONDecodeError:
            raise ValueError(f"'{config_path}' 파일의 JSON 형식이 올바르지 않습니다.")
        
        self._initialized = True

    @property
    def path(self):
        return self._config_data.get('path', {})

    @property
    def hyperparameter(self):
        return self._config_data.get('hyperparameter', {})

    @property
    def library(self):
        return self._config_data.get('library', {})

    def get_all_config(self):
        return self._config_data