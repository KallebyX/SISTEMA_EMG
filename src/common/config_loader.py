import json
import yaml

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
