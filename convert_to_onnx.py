from model import Classifire
import torch

@torch.no_grad()
def main():
    classifire = Classifire.load_from_checkpoint("model.ckpt")
    classifire.eval()
    input_data = torch.zeros((1, 2, 3, 64, 64))
    classifire.to_onnx("model.onnx", 
                input_data,
                export_params=True,
                input_names = ['input'],  
                output_names = ['output'], 
                dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

if __name__ == "__main__":
    main()