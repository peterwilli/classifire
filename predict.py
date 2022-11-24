from model import Classifire
import sys
import torch
import torchvision
from torchvision import transforms
import random
from PIL import Image
from PIL import ImageOps
import os
import numpy as np

pred_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

@torch.no_grad()
def images_from_paths(paths):
    images = None
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = image.resize((64, 64))
        image = pred_transforms(np.array(image)).unsqueeze(0)
        if images is None:
            images = image
        else:
            images = torch.cat((images, image), 0)
    return images

@torch.no_grad()
def main():
    if len(sys.argv) < 3:
        print("Error: needs 2 image files (one as source one to detect)")
        return

    images = images_from_paths(sys.argv[1:3])
    images = images.unsqueeze(0) # Simulate batch
    print("images", images.shape)

    classifire = Classifire.load_from_checkpoint("model.ckpt")
    classifire.eval()

    logits = classifire(images)
    print(f"logits: {logits}")
    preds = torch.argmax(logits, dim=1)
    is_detected = preds == 1
    print(f"Detected same object: {is_detected}")

if __name__ == "__main__":
    main()