import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from glob import glob
import time

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

model = UNet()
print(model)

class ImageEnhancementDataset(Dataset):
    def __init__(self, raw_image_dir, enhanced_image_dir, transform=None):
        self.raw_image_paths = sorted(glob(os.path.join(raw_image_dir, '*.jpg')))
        self.enhanced_image_paths = sorted(glob(os.path.join(enhanced_image_dir, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.raw_image_paths)

    def __getitem__(self, idx):
        raw_image = Image.open(self.raw_image_paths[idx]).convert('RGB')
        enhanced_image = Image.open(self.enhanced_image_paths[idx]).convert('RGB')
        if self.transform:
            raw_image = self.transform(raw_image)
            enhanced_image = self.transform(enhanced_image)
        return raw_image, enhanced_image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

raw_image_dir = './raw'
enhanced_image_dir = './output'
dataset = ImageEnhancementDataset(raw_image_dir, enhanced_image_dir, transform=transform)

# Use a smaller subset of the dataset
train_size = int(0.1 * len(dataset))  # 10% of the dataset
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5  # Reduce number of epochs for quicker testing
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Batch [{i}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
    
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}, Time: {epoch_time:.2f}s")

print("Training complete.")

# Save the model
model_path = 'image_enhancement_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load the model (for inference)
model.load_state_dict(torch.load(model_path))
model.eval()

# Example inference on a new image
def enhance_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        enhanced_image = model(image)
        enhanced_image = nn.functional.interpolate(enhanced_image, size=(224, 224), mode='bilinear', align_corners=False)
        enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).numpy()
        enhanced_image = (enhanced_image * 255).astype(np.uint8)
        enhanced_image = Image.fromarray(enhanced_image)
        return enhanced_image

# Use the model to enhance a new image
new_image_path = 'test.jpg'
enhanced_image = enhance_image(new_image_path, model)
enhanced_image.save('enhanced_new_image.jpg')
print("Enhanced image saved as 'enhanced_new_image.jpg'")
