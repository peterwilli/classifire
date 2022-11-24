import yaml
from yaml import Loader, Dumper
import sys
from PIL import Image
from PIL import ImageOps
import os
import base64
from io import BytesIO

def image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def images_from_paths(paths):
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = image.resize((64, 64))
        images.append(image)
    return images

if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_path, "base_flow.yml"), "r") as f:
        base_flow = yaml.safe_load(f.read())
    environment_variables = {}
    image_paths = sys.argv[1:]
    images = images_from_paths(image_paths)
    for idx, image in enumerate(images):
        environment_variables[f"CLASSIFIRE_INPUT_{idx}"] = image_to_base64(image)
    base_flow["executors"][0]["env"] = environment_variables
    print(yaml.dump(base_flow, sort_keys=False))