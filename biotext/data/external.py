import os
import yaml
from pathlib import Path
from typing import Dict


class Config:
    "Create config for save/load of datasets and models"
    config_path = Path(os.getenv('BIOTEXT_HOME', "~/.biotext")).expanduser()
    config_file = config_path/'config.yml'

    def __init__(self):
        self.config_path.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists():
            self.create_config()
        self.cfg = self.load_config()

    def __getitem__(self,k):
        k = k.lower()
        if k not in self.cfg: k = f"path_{k}"
        return Path(self.cfg[k])

    def create_config(self) -> None:
        "Create new config with default paths"
        config = {
            'path_data': str(self.config_path/'data'),
            'path_storage': str(self.config_path/'tmp'),
            'path_model': str(self.config_path/'models'),
            'path_archive': str(self.config_path/'archive')
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

def path(fname: str = '.', c_key: str = 'archive') -> Path:
    "Return local path where to download based on `c_key`"
    local_path = Path.cwd()/('models' if c_key=='models' else 'data')/fname
    if local_path.exists(): return local_path
    return Config()[c_key]/fname

if __name__ == "__main__":
    Config()
    print(path(fname='wikitext-103', c_key='data'))
