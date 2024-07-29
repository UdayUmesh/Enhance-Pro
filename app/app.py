import os
import shutil
import torch
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from torchvision import transforms

app = Flask(__name__)

# Ensure these paths match your directory structure
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the ONNX model
quantized_model_path = 'models/image_enhancement_quantized.onnx'
model = onnx.load(quantized_model_path)

# Create ONNX Runtime session options
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Configure Vitis AI execution provider
config_file_path = "vaip_config.json"
aie_session = ort.InferenceSession(
    quantized_model_path, 
    sess_options=sess_options,
    providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
    provider_options=[{'config_file': config_file_path, 'cacheDir': 'cache', 'cacheKey': 'hello_cache'}, {}]
)

def enhance_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0).numpy()

    ipu_results = aie_session.run(None, {'input': input_image})

    enhanced_image = ipu_results[0][0]
    enhanced_image = np.transpose(enhanced_image, (1, 2, 0))  # Convert to HWC format
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    enhanced_image_pil = Image.fromarray(enhanced_image)
    
    enhanced_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced_image.jpg')
    enhanced_image_pil.save(enhanced_image_path)

    return enhanced_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = 'uploaded_image.jpg'
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            
            enhanced_image_path = enhance_image(image_path)
            
            return render_template('index.html', original_image=filename, enhanced_image=os.path.basename(enhanced_image_path))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
