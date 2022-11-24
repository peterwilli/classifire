from jina import Executor, requests, DocumentArray, Document
import os
import onnxruntime as ort
import torchvision
from torchvision import transforms
import torch
from io import BytesIO
import base64
from PIL import Image
from PIL import ImageOps
import numpy as np

ort_session = ort.InferenceSession("./model.onnx")

tensor_to_pil_image = transforms.ToPILImage()
pred_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

original_images = []
for key, value in os.environ.items():
    if key.startswith("CLASSIFIRE_INPUT_"):
        image = Image.open(BytesIO(base64.b64decode(value)))
        image = pred_transforms(np.array(image)).unsqueeze(0)
        original_images.append(image)

def argavg(logits):
    is_detected = logits[1] > logits[0]
    if is_detected:
        result = logits[0] / logits[1]
    else:
        result = logits[1] / logits[0]
    
    return is_detected, result

def inference(img0, img1):
    images = torch.cat((img0, img1), 0)
    images = images.unsqueeze(0)
    ort_inputs = { ort_session.get_inputs()[0].name: images.numpy() }
    logits = ort_session.run(None, ort_inputs)[0][0]
    (is_detected, how_much) = argavg(logits)
    print(f"is_detected: {is_detected} ({how_much:.4f}%)")
    return is_detected

@torch.no_grad()
def images_from_docs(docs):
    images = []
    for doc in docs:
        doc.load_uri_to_image_tensor()
        image = tensor_to_pil_image(doc.tensor).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = image.resize((64, 64))
        image = pred_transforms(np.array(image)).unsqueeze(0)
    images.append(image)
    return images

class ClassifireExecutor(Executor):
    @requests
    def classify(self, docs, **kwargs):
        input_images = images_from_docs(docs)
        for input_image in input_images:
            for idx, original_image in enumerate(original_images):
                if inference(input_image, original_image):
                    return DocumentArray(Document(text="match", tags={"match_index": idx}))
        return DocumentArray(Document(text="no_match"))