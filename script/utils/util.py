import json
import os

# Load configuration
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Construct dynamic paths
    paths = {
        key: os.path.join(config["base_paths"]["output"], filename) 
        for key, filename in config["files"].items()
    }
    
    # Add base paths separately
    paths["input_path"] = config["base_paths"]["input"]
    paths["output_path"] = config["base_paths"]["output"]

    return paths
