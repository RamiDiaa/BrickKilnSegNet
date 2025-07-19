#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from glob import glob
import rasterio
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as T
import os

# Configuration
MODEL_PATH = 'best_model_ce_tv_d13_m07_y2025_h16_m56_s23.pth'
NUM_CLASSES = 11
NUM_CHANNELS = 8
BATCH_SIZE = 32
ENCODER_NAME = "resnext101_32x8d"
ENCODER_WEIGHTS = "imagenet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = 'results'

# Use training set statistics (DO NOT recompute on test data)
GLOBAL_MEANS = [0.03402944654226303, 0.04915359988808632, 0.056084536015987396,
                0.1244724690914154, 0.12229487299919128, 0.09260836988687515,
                0.17983973026275635, -0.011018575169146061]


print("=== Prediction Configuration ===")
print(f"{'Number of Classes:':<20} {NUM_CLASSES}")
print(f"{'Input Channels:':<20} {NUM_CHANNELS}")
print(f"{'Batch Size:':<20} {BATCH_SIZE}")
print(f"{'Encoder Name:':<20} {ENCODER_NAME}")
print(f"{'Device:':<20} {DEVICE}")

# Image normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

# Data handling classes
class FillInvalid:
    def __init__(self, global_means, last_k=2):
        self.global_means = global_means
        self.last_k = last_k

    def __call__(self, image):
        img = torch.as_tensor(image).float()
        invalid = ~torch.isfinite(img)
        C = img.shape[0]

        for c in range(C):
            mask_c = invalid[c]
            if not mask_c.any():
                continue

            if c >= C - self.last_k:
                med = torch.nanmedian(img[c])
                img[c][mask_c] = med
            else:
                img[c][mask_c] = self.global_means[c]

        return img  # Return tensor directly

class PredictionDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.profiles = []  # Store geospatial metadata
        
        # Preload profiles (metadata only)
        for path in tqdm(img_paths, desc="Loading metadata"):
            with rasterio.open(path) as src:
                self.profiles.append(src.profile)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as src:
            image = src.read().astype(np.float32)
            image_tensor = torch.from_numpy(image).float()
            image_tensor[:3] = normalize(image_tensor[:3])
        
        if self.transform:
            image_tensor = self.transform(image_tensor.numpy())
            
        return image_tensor, os.path.basename(self.img_paths[idx]), idx

# Model setup
model = smp.UnetPlusPlus(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=NUM_CHANNELS,
    classes=NUM_CLASSES, 
    activation=None
).to(DEVICE)

# Load trained model
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully")

# Prediction function
def predict(model, loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, filenames, idxs in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8)
            
            for i, (pred, filename, idx) in enumerate(zip(preds, filenames, idxs)):
                profile = loader.dataset.profiles[idx]
                output_path = os.path.join(output_dir, f"pred_{filename}")
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    profile.update(
                        dtype=rasterio.uint8,
                        count=1,
                        compress='lzw'
                    )
                    dst.write(pred, 1)

# Get prediction images
img_tif_files = glob('Test_image_case_study\\Image\\*.tif', recursive=True)
img_tif_files.sort()
print(f"Found {len(img_tif_files)} images for prediction")

# Create dataset and loader
fill_invalid = FillInvalid(GLOBAL_MEANS)
pred_dataset = PredictionDataset(img_tif_files, transform=fill_invalid)
pred_loader = DataLoader(
    pred_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=0
)

# Run prediction
predict(model, pred_loader, DEVICE, OUTPUT_DIR)
print(f"Predictions saved to: {OUTPUT_DIR}")