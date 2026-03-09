import json


def load_json(path):
    try:
        with open(path, 'r') as f:
            file = json.load(f)
    except Exception:
        print(f"Couldn't load {path}")
        return None
        
    return file