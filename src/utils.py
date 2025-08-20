import os, yaml
from safetensors.torch import save_file

def load_config(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def save_safetensors(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(state_dict, path)
