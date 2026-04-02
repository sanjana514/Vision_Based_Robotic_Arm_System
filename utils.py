import json

def load_metrics():
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
    except:
        # default data jodi file na thake
        data = {
            "fps": 0,
            "success": 0,
            "retry": 0,
            "avg_pick": 0,
            "energy": 0,
            "ppm": 0
        }
    return data