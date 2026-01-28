import json


def load_json(path):
    try:
        with open(path, 'r') as f:
            file = json.load(f)
        pass
    except:
        print(f"Couldn't load {path}")
        return None
        
    return file