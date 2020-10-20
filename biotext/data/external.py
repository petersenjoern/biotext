import os
import yaml
from pathlib import Path
from typing import Dict


class Config:
    "Create config for save/load of datasets and models"
    config_path = Path(os.getenv('BIOTEXT_HOME', "~/.biotext")).expanduser()
    config_file = config_path.joinpath('config.yml')

    def __init__(self):
        self.config_path.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists():
            self.create_config()
        self.cfg = self.load_config()

    def create_config(self) -> None:
        "Create new config with default paths"
        config = {
            'path_data': str(self.config_path.joinpath('data')),
            'path_storage': str(self.config_path.joinpath('tmp')),
            'path_model': str(self.config_path.joinpath('models'))
        }
        self.save_file(config)

    def save_file(self, config: Dict) -> None:
        "Save config file with default config location at ~/.biotext"
        with self.config_file.open('w') as f: yaml.dump(config, f, default_flow_style=False)

    def load_config(self) -> Dict:
        "Load and return config"
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            return config

if __name__ == "__main__":
    Config()
