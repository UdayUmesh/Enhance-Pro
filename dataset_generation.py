import cv2
import os
from glob import glob

# Define the auto_tone function
def auto_tone(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# Path to the folder containing raw images
input_folder = './raw'
output_folder = './output'
os.makedirs(output_folder, exist_ok=True)

# Process each image
for img_path in glob(os.path.join(input_folder, '*.jpg')):
    print(f"Processing {img_path}")
    img = cv2.imread(img_path)
    
    # Check if image is loaded
    if img is None:
        print(f"Error: Unable to load image at {img_path}. Skipping this file.")
        continue
    
    enhanced_img = auto_tone(img)
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, enhanced_img)
    print(f"Saved enhanced image to {output_path}")
