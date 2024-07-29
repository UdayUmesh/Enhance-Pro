import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from glob import glob
import time
import onnx
import onnxruntime as ort
import shutil
from timeit import default_timer as timer
import vai_q_onnx
import numpy as np

# Define the UNet Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up(e3)
        d3 = self.crop_and_concat(d3, e2)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = self.crop_and_concat(d2, e1)
        d2 = self.dec2(d2)
        d1 = self.dec1(self.up(d2))
        return d1
    
    def crop_and_concat(self, upsampled, bypass):
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]
        upsampled = nn.functional.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat((upsampled, bypass), 1)

# Load your trained model
model = UNet()
model.load_state_dict(torch.load('image_enhancement_model.pth', map_location=torch.device('cpu')))
model.eval()

# Dummy input for ONNX export
dummy_input = torch.randn(1, 3, 224, 224)

os.makedirs("models", exist_ok=True)

# Export the model to ONNX format
tmp_model_path = "models/image_enhancement.onnx"
torch.onnx.export(
    model,
    dummy_input,
    tmp_model_path,
    export_params=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Model exported to ONNX format.")

# Quantize the Model
input_model_path = "models/image_enhancement.onnx"
output_model_path = "models/image_enhancement_quantized.onnx"

vai_q_onnx.quantize_static(
    input_model_path,
    output_model_path,
    calibration_data_reader=None,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_ipu_cnn=True,
    extra_options={'ActivationSymmetric': True}
)
print('Calibrated and quantized model saved at:', output_model_path)

# Load the quantized model
quantized_model_path = 'models/image_enhancement_quantized.onnx'
model = onnx.load(quantized_model_path)

# Create ONNX Runtime session options
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Load and preprocess the image
image_path = 'test.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_image = transform(image).unsqueeze(0).numpy()

# Run the Model on CPU
cpu_session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'], sess_options=sess_options)
start = timer()
cpu_results = cpu_session.run(None, {'input': input_image})
cpu_total = timer() - start

# Post-process and save CPU output image
cpu_enhanced_image = cpu_results[0].squeeze(0).transpose(1, 2, 0)
cpu_enhanced_image = (cpu_enhanced_image * 255).astype(np.uint8)
cpu_enhanced_image = Image.fromarray(cpu_enhanced_image)
cpu_enhanced_image.save('enhanced_image_cpu.jpg')
print(f"Enhanced image saved as enhanced_image_cpu.jpg")
print(f"CPU Execution Time: {cpu_total}")

# Create cache directory if it doesn't exist
current_directory = os.getcwd()
cache_directory = os.path.join(current_directory, 'cache')
if os.path.exists(cache_directory):
    shutil.rmtree(cache_directory)
os.makedirs(cache_directory)

# Configure Vitis AI execution provider
config_file_path = "vaip_config.json"
aie_options = ort.SessionOptions()
aie_session = ort.InferenceSession(model.SerializeToString(), providers=['VitisAIExecutionProvider'], sess_options=aie_options,
                                   provider_options=[{'config_file': config_file_path, 'cacheDir': cache_directory, 'cacheKey': 'hello_cache'}])

# Run the Model on IPU
start = timer()
ipu_results = aie_session.run(None, {'input': input_image})
ipu_total = timer() - start

# Post-process and save IPU output image
ipu_enhanced_image = ipu_results[0].squeeze(0).transpose(1, 2, 0)
ipu_enhanced_image = (ipu_enhanced_image * 255).astype(np.uint8)
ipu_enhanced_image = Image.fromarray(ipu_enhanced_image)
ipu_enhanced_image.save('enhanced_image_ipu.jpg')
print(f"Enhanced image saved as enhanced_image_ipu.jpg")
print(f"IPU Execution Time: {ipu_total}")
