import os
import torch
import torch.nn as nn
from torchvision import models

def export_to_onnx():
    model_pt_path = "app/models/cnn_model_v1.pt"
    model_onnx_path = "app/models/cnn_model_v1.onnx"
    
    if not os.path.exists(model_pt_path):
        print(f"Error: Trained PyTorch weights not found at {model_pt_path}")
        return
        
    print("Loading PyTorch model weights...")
    device = torch.device("cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_pt_path, map_location=device))
    model.eval()
    
    print("Creating dummy input tensor...")
    # ResNet18 expects 3-channel 224x224 input
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    
    print("Exporting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        model_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    if os.path.exists(model_onnx_path):
        print(f"Success! Model successfully exported to ONNX format at {model_onnx_path}")
    else:
        print("Error: Export failed.")

if __name__ == "__main__":
    export_to_onnx()
