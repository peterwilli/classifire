import numpy as np
import math
import gradio as gr
import onnxruntime as ort
from PIL import Image
from PIL import ImageOps
import torchvision
from torchvision import transforms
import torch
from imgaug import augmenters as iaa

pred_transforms = transforms.Compose(
    [
        iaa.Resize({"shorter-side": 64, "longer-side": 64}).augment_image,
        iaa.CropToFixedSize(width=64, height=64).augment_image,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

ort_session = ort.InferenceSession("../../model.onnx")

@torch.no_grad()
def images_from_paths(paths):
    images = None
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = pred_transforms(np.array(image)).unsqueeze(0)
        if images is None:
            images = image
        else:
            images = torch.cat((images, image), 0)
    return images

def argavg(logits):
    print(logits)
    is_detected = logits[1] > logits[0]
    if is_detected:
        result = logits[0] / logits[1]
    else:
        result = logits[1] / logits[0]
    
    return is_detected, result

def inference(img0, img1):
    paths = [img0, img1]
    images = images_from_paths(paths)
    images = images.unsqueeze(0) # Simulate batch
    ort_inputs = { ort_session.get_inputs()[0].name: images.numpy() }
    logits = ort_session.run(None, ort_inputs)[0][0]
    (is_detected, how_much) = argavg(logits)
    print(f"is_detected: {is_detected} ({how_much:.4f}%)")
    return is_detected

def image_grid(imgs, cols):
    cols = min(cols, len(imgs))
    rows = math.ceil(len(imgs) / cols)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def render_group(group) -> Image:
    img_size = 256
    cluster_img = Image.new(mode = "RGB",
                    size = (img_size, img_size),
                    color = (255, 153, 255))
    for idx, img_path in enumerate(group):
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        img = ImageOps.contain(img, (img_size, img_size))
        cluster_img.paste(img, (idx * math.ceil(img_size * 0.05), idx * math.ceil(img_size * 0.05)))
    return cluster_img

def render_cluster(cluster) -> Image:
    cluster_len = len(cluster)
    groups = []
    for group in cluster:
        group_img = render_group(group)
        groups.append(group_img)
    return image_grid(groups, 2)

if __name__ == '__main__':
    with gr.Blocks() as demo:
        cluster_state = gr.State([])
        gr.Markdown("Flip text or image files using this demo.")
        with gr.Tab("Detect if images are same class"):
            with gr.Row():
                img_1 = gr.Image(label="Input Image", type="filepath")
                img_2 = gr.Image(label="Match Image", type="filepath")
            detect_class_output = gr.Textbox(label="Has same class as input")
            btn_check_match = gr.Button("Check")

        with gr.Tab("Cluster images"):
            error_box = gr.Textbox(label="Error", visible=False)
            with gr.Row():
                image_input = gr.Image(type="filepath")
                image_button = gr.Button("Add to cluster")
            image_output = gr.Image()

        def inference_cluster(img0, cluster):
            if img0 is None:
                return {error_box: gr.update(value="Please provide an image to add to the cluster!", visible=True)}
            if len(cluster) == 0:
                cluster.append([img0])
            else:
                is_found = False
                for group in cluster:
                    first_image = group[0]
                    if inference(first_image, img0):
                        group.append(img0)
                        is_found = True
                        break
                    if is_found:
                        break
                    
                if not is_found:
                    cluster.append([img0])
            cluster_img = render_cluster(cluster)
            return {image_input: gr.update(value=None), cluster_state: cluster, image_output: cluster_img, error_box: gr.update(value="", visible=False)}

        btn_check_match.click(inference, inputs=[img_1, img_2], outputs=detect_class_output)
        image_button.click(inference_cluster, inputs=[image_input, cluster_state], outputs=[error_box, image_output, image_input, cluster_state])

demo.launch()