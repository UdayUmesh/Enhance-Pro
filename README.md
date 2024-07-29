AMD developer Contest

This project involved creating an AI-based image enhancement web application utilizing a UNet model trained on a dataset of raw and auto-enhanced images. The steps included:

Dataset Preparation:

Raw Images: Original images requiring enhancement.(raw - [sample_raw_images}))
Enhanced Images: Images processed using CLAHE for better quality(output - [sample_enhanced_images_for training]).
Model Training:

Architecture: A custom UNet model was employed.
Training: The model was trained using paired raw and enhanced images, optimizing with MSE loss and Adam optimizer.
Model Deployment:

ONNX Conversion: The trained model was converted to ONNX format.
Quantization: The ONNX model was quantized for efficient inference.
IPU Integration: Inference was run on the IPU using Vitis AI Execution Provider.
Web Application:

Flask: Built a Flask-based web interface for users to upload images.
Enhancement: Uploaded images are enhanced using the IPU-accelerated model.
UI: A clean and user-friendly interface for image upload and display.
The application enhances user-uploaded images by processing them through a trained UNet model running on an IPU, providing an efficient and improved image enhancement experience.

![image](https://github.com/user-attachments/assets/5dcf2bcb-f968-404a-810f-f027f080490e)

![image](https://github.com/user-attachments/assets/8a777ad1-cd72-44fe-a4f6-b16c9be9d9a5)


![WhatsApp Image 2024-07-29 at 10 31 48_01ab91dc](https://github.com/user-attachments/assets/203f6e02-e643-4f14-9cad-383fd846c1f9)
