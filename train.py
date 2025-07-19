#!/usr/bin/env python
# coding: utf-8

# In[74]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
# import matplotlib.pyplot as plt
# import torchvision.utils as vutils
import numpy as np
import segmentation_models_pytorch as smp
from glob import glob
import rasterio
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import label
from sklearn.metrics import f1_score, precision_recall_fscore_support
import albumentations as A
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from collections import defaultdict
# from focal_loss import sparse_categorical_focal_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T





# In[75]:


MODEL_CODE = datetime.now().strftime("d%d_m%m_y%Y_h%H_m%M_s%S")
NUM_CLASSES = 11
NUM_CHANNELS = 8

BATCH_SIZE = 32

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

N_EPOCH = 100
VAL_RATIO = 0.3


ENCODER_NAME = "resnext101_32x8d"
ENCODER_WEIGHTS = "imagenet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=== Training Configuration ===")
print(f"{'Model Code:':<20} {MODEL_CODE}")
print(f"{'Number of Classes:':<20} {NUM_CLASSES}")
print(f"{'Input Channels:':<20} {NUM_CHANNELS}")
print(f"{'Batch Size:':<20} {BATCH_SIZE}")
print(f"{'Learning Rate:':<20} {LEARNING_RATE}")
print(f"{'optimizer weight decay:':<20} {WEIGHT_DECAY}")
print(f"{'Epochs:':<20} {N_EPOCH}")
print(f"{'Validation Ratio:':<20} {VAL_RATIO}")
print(f"{'Encoder Name:':<20} {ENCODER_NAME}")
print(f"{'Encoder Pre Trained Weights:':<20} {ENCODER_WEIGHTS}")
print(f"{'loss function':<20} ce + tversky a0.3 b0.7 g1.75 combined 0.05ce 0.95tv")
print(f"{'model':<20} unet ++")

# 

# In[76]:


# img_tif_files = glob('Brick_Data_Train\\Brick_Data_Train\\Image\\*.tif', recursive=True)
# mask_tif_files = glob('Brick_Data_Train\\Brick_Data_Train\\Mask\\*.tif', recursive=True)

# the paths on palma
img_tif_files = glob('/scratch/tmp/Brick_Data_Train/Image/*.tif', recursive=True)
mask_tif_files = glob('/scratch/tmp/Brick_Data_Train/Mask/*.tif', recursive=True)

img_tif_files.sort()
mask_tif_files.sort()


print(f"Total image files: {len(img_tif_files)}")
print(f"Total mask files: {len(mask_tif_files)}")

DATA_COUNT= len(img_tif_files)


# In[77]:


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

class BrickKilnDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image and mask as numpy
        with rasterio.open(self.img_paths[idx]) as src:
            image = src.read().astype(np.float32)
            image_tensor = torch.from_numpy(image).float()  # Full image if needed
            # image[:3,:,:] = normalize(torch.from_numpy(image[:3,:,:]))
            image_tensor[:3] = normalize(image_tensor[:3])
            image = image_tensor.numpy()


        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)

        # Apply your fill-invalid transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Now ensure both are torch.Tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        # else, assume it's already a torch.Tensor

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        # else, assume torch.Tensor

        return image, mask


# In[78]:


# Load dataset without transform to compute stats
raw_dataset = BrickKilnDataset(img_tif_files, mask_tif_files)

# Dataloader to iterate through all images
stats_loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[79]:


def calculate_class_mask_stats(loader):
    class_counts = defaultdict(int)
    total_pixels = 0

    for _, masks in tqdm(loader, desc="Calculating class pixel stats"):
        # If masks is a batch, iterate over it
        if masks.ndim == 4:  # (B, C, H, W) or (B, 1, H, W)
            masks = masks.squeeze(1)  # make it (B, H, W)
        elif masks.ndim == 3 and masks.shape[0] > 1:  # (B, H, W)
            pass
        else:
            masks = masks.unsqueeze(0)  # single mask, add batch dimension

        for mask in masks:
            unique_classes, counts = torch.unique(mask, return_counts=True)
            for cls, cnt in zip(unique_classes.tolist(), counts.tolist()):
                class_counts[cls] += cnt
            total_pixels += mask.numel()

    print(f"Total pixels: {total_pixels:,}\n")
    print("Class-wise pixel counts and percentages:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_pixels) * 100 if total_pixels else 0
        print(f"Class {cls}: {count:,} pixels ({percentage:.2f}%)")

    return dict(class_counts), total_pixels

# calculate_class_mask_stats(stats_loader)

"""

Total pixels: 489,619,456

Class-wise pixel counts and percentages:
Class 0: 489,273,929 pixels (99.93%)
Class 1: 110,363 pixels (0.02%)
Class 2: 166,043 pixels (0.03%)
Class 3: 24,800 pixels (0.01%)
Class 4: 7,167 pixels (0.00%)
Class 5: 12,314 pixels (0.00%)
Class 6: 3,204 pixels (0.00%)
Class 7: 2,145 pixels (0.00%)
Class 8: 19,491 pixels (0.00%)

"""


# In[80]:


def compute_invalid_percentages(loader):
    nan_counts = None
    inf_counts = None
    total_pixels = 0
    for imgs, _ in loader:
        # imgs shape: [B, C, H, W]
        B, C, H, W = imgs.shape
        if nan_counts is None:
            nan_counts = torch.zeros(C, dtype=torch.long)
            inf_counts = torch.zeros(C, dtype=torch.long)
        total_pixels += B * H * W

        # Count NaNs
        nan_counts += torch.isnan(imgs).sum(dim=[0, 2, 3])
        # Count Infs
        inf_counts += torch.isinf(imgs).sum(dim=[0, 2, 3])

    pct_nan = nan_counts.float() / (total_pixels / C) * 100
    pct_inf = inf_counts.float() / (total_pixels / C) * 100

    for i, (n_pct, i_pct) in enumerate(zip(pct_nan, pct_inf)):
        print(f"Channel {i}: {n_pct:.2f}% NaN, {i_pct:.2f}% Inf")

    return pct_nan, pct_inf

# Run on training set
# gpct_nan, gpct_inf = compute_invalid_percentages(stats_loader)


"""
Channel 0: 0.00% NaN, 0.00% Inf
Channel 1: 0.00% NaN, 0.00% Inf
Channel 2: 0.00% NaN, 0.00% Inf
Channel 3: 0.00% NaN, 0.00% Inf
Channel 4: 0.00% NaN, 0.00% Inf
Channel 5: 0.00% NaN, 0.00% Inf
Channel 6: 0.79% NaN, 0.00% Inf
Channel 7: 0.78% NaN, 0.00% Inf

"""


# In[81]:


# compute per‐channel global mean & per‐image medians for last 2 channels


"""
sum_vals = torch.zeros(NUM_CHANNELS)
count_vals = torch.zeros(NUM_CHANNELS)

for imgs, _ in tqdm(stats_loader, desc="Computing global means"):
    valid = torch.isfinite(imgs)
    sum_vals   += torch.where(valid, imgs, 0.0).sum(dim=[0,2,3])
    count_vals += valid.sum(dim=[0,2,3]).float()

global_means = (sum_vals / count_vals).tolist()
"""

global_means = [0.03402944654226303, 0.04915359988808632, 0.056084536015987396,
                0.1244724690914154, 0.12229487299919128, 0.09260836988687515,
                0.17983973026275635, -0.011018575169146061]
# global_means = NUM_CHANNELS * [0] #try this
print("Global channel means:", global_means)


# In[82]:


class FillInvalid:
    def __init__(self, global_means, last_k=2):
        self.global_means = global_means
        self.last_k = last_k

    def __call__(self, image=None, mask=None, **kwargs):
        # Convert to torch tensor
        img = torch.as_tensor(image).float()

        invalid = ~torch.isfinite(img)
        C = img.shape[0]

        for c in range(C):
            mask_c = invalid[c]
            if not mask_c.any():
                continue

            # print(f"Channel {c}: {mask_c.sum().item()} NaN/Inf values replaced")

            if c >= C - self.last_k:
                med = torch.nanmedian(img[c])
                img[c][mask_c] = med
            else:
                img[c][mask_c] = self.global_means[c]

        return {'image': img, 'mask': mask}
    



# In[83]:


class CombinedTransform:
    def __init__(self, fill_invalid, albumentations_transform=None):
        self.fill_invalid = fill_invalid
        self.albumentations_transform = albumentations_transform

    def __call__(self, image, mask):
        result = self.fill_invalid(image=image, mask=mask)
        img_tensor = result['image']
        mask = result['mask']

        if self.albumentations_transform:
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask

            augmented = self.albumentations_transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']

            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask_np).long()

        return {'image': img_tensor, 'mask': mask}


# In[84]:


albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.1),
    A.RandomRotate90(p=0.1),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-45, 45), p=0.1),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.1),
], additional_targets={'mask': 'mask'})


# In[85]:


fill_invalid = FillInvalid(global_means, last_k=2)
combined_transform = CombinedTransform(fill_invalid, albumentations_transform)

# TRAIN transform (with augmentation)
train_transform = CombinedTransform(
    fill_invalid=FillInvalid(global_means),
    albumentations_transform=None
)

# VALIDATION transform (no augmentation)
val_transform = CombinedTransform(
    fill_invalid=FillInvalid(global_means),
    albumentations_transform=None
)




# In[86]:


"""
props_binary = []
for img_batch, mask_batch in tqdm(stats_loader, desc="splitting indices", leave=False):
    for mask in mask_batch:
        mask_np = mask.numpy().ravel().astype(int)
        counts = np.bincount(mask_np, minlength=NUM_CLASSES)
        # binary indicator: does this image contain *any* pixels of class i?
        present = (counts > 0).astype(int)
        props_binary.append(present)

props_binary = np.stack(props_binary, axis=0)  # shape = [N_images, NUM_CLASSES]

# Drop the background column (class 0), since it's always present
y = props_binary[:, 1:]   # now shape = [N_images, NUM_CLASSES-1]

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=VAL_RATIO, random_state=42
)

train_indices, val_indices = next(msss.split(X=np.zeros(len(y)), y=y))

"""

all_indices = np.arange(DATA_COUNT)
np.random.shuffle(all_indices)

# Split indices
val_size = int(VAL_RATIO * DATA_COUNT)
val_indices = all_indices[:val_size]
train_indices = all_indices[val_size:]

np.save('val_indices_{}.npy'.format(MODEL_CODE), val_indices)

# In[87]:


# 2. Create datasets with different transforms
train_dataset = BrickKilnDataset(
    img_paths=[img_tif_files[i] for i in train_indices],
    mask_paths=[mask_tif_files[i] for i in train_indices],
    transform = train_transform
)

val_dataset = BrickKilnDataset(
    img_paths=[img_tif_files[i] for i in val_indices],
    mask_paths=[mask_tif_files[i] for i in val_indices],
    transform = val_transform

)

# 3. Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[88]:


images, masks = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")  # Expected: (B, C, H, W)
print(f"Mask batch shape: {masks.shape}")    # Expected: (B, H, W)
# for i in range(images.shape[1]):
#     band = images[1, i]
#     print(f"Band {i} max: {band.max().item()}")



# In[89]:


def calculate_class_weights(dataloader, num_classes):
    pixel_counts = np.zeros(num_classes, dtype=np.float64)

    # 1) accumulate counts
    for _, masks in dataloader:

        counts = np.bincount(masks.cpu().numpy().ravel(), minlength=num_classes)
        pixel_counts += counts

    # 2) compute frequencies
    total = pixel_counts.sum()
    freq = pixel_counts / total

    # 3) compute median over nonzero classes
    nonzero = freq > 0
    median = np.median(freq[nonzero])

    # 4) weights & optional normalization
    weights = median / np.where(freq == 0, 1e-6, freq)
    # e.g. normalize so sum weights = num_classes:
    weights = weights * num_classes / weights.sum()

    return weights




# weights = calculate_class_weights(train_loader, NUM_CLASSES)

weights = [ 4.35963610e-06, 1.89684419e-02, 1.24808249e-02, 8.69165799e-02, 
            2.77582641e-01, 1.75237343e-01, 6.77213522e-01, 9.18832417e-01, 
            1.19743059e-01, 4.35651041e+00, 4.35651041e+00]

print("Calculated class weights:", weights)



# In[ ]:


def evaluate_pixel_based(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0

    all_preds = []
    all_masks = []
    
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, masks in progress_bar:
            inputs = inputs.to(device)
            masks = masks.long().to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    # Concatenate all predictions and masks
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_masks)

    # Convert to numpy for sklearn
    y_true_np = y_true.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()

    # Global pixel accuracy (includes background)
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_pixels

    # Identify non-background classes present
    non_bg_classes = np.setdiff1d(
        np.unique(np.concatenate([y_true_np, y_pred_np])),
        [0]
    )

    # Calculate weighted F1 (IGNORES class 0 but counts errors involving it)
    f1_weighted = 0.0
    per_class_metrics = []
    
    if len(non_bg_classes) > 0:
        f1_weighted = f1_score(
            y_true_np, 
            y_pred_np, 
            labels=non_bg_classes,
            average='weighted',
            zero_division=0
        )
        
        # Get per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, 
            y_pred_np, 
            labels=non_bg_classes,
            zero_division=0
        )
        
        per_class_metrics = list(zip(non_bg_classes, precision, recall, f1, support))

    # Reporting
    print(f"\nValidation Loss: {avg_loss:.4f}, Pixel Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 : {f1_weighted:.4f}\n")
    
    if per_class_metrics:
        print(f"{'Class':<8}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
        print("-" * 48)
        for cls, p, r, f, s in per_class_metrics:
            print(f"{cls:<8}{p:>10.4f}{r:>10.4f}{f:>10.4f}{s:>10}")


    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]    
    without_bg_f1_weighted = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    print(f"\n old f1 score(excluding background false positives): {without_bg_f1_weighted:.4f}\n")


    # Treat non-zero classes as a single class (1) for binary metrics
    y_true_binary = (y_true_np != 0).astype(int)
    y_pred_binary = (y_pred_np != 0).astype(int)
    
    # Calculate binary metrics in single call
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        y_true_binary, 
        y_pred_binary, 
        average='binary',
        zero_division=0
    )
    
    print(f"\nBinary Metrics (non-zero classes as one):")
    print(f"Precision: {binary_precision:.4f}, Recall: {binary_recall:.4f}, F1: {binary_f1:.4f}")


    return avg_loss, accuracy, f1_weighted


# In[91]:


def save_checkpoint(model, optimizer, epoch, loss, batch_size, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'batch_size': batch_size,
    }, filename)
    print(f"Checkpoint saved at epoch {epoch}")


# In[92]:


model = smp.UnetPlusPlus(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=NUM_CHANNELS,
    classes=NUM_CLASSES, 
    activation=None
).to(DEVICE)

# # 1) Freeze everything
# for param in model.parameters():
#     param.requires_grad = False

# # 2) Un‑freeze conv1 (and its BN)
# for param in model.encoder.conv1.parameters():
#     param.requires_grad = True
# for param in model.encoder.bn1.parameters():
#     param.requires_grad = True

# # 3) Un‑freeze the entire decoder
# for param in model.decoder.parameters():
#     param.requires_grad = True

# # 4) Un‑freeze the segmentation head (your final 1×1 conv + activation)
# for param in model.segmentation_head.parameters():
#     param.requires_grad = True


# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)

weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
#loss_fn = nn.CrossEntropyLoss(weight = weights_tensor)
# loss_fn = sparse_categorical_focal_loss(class_weight=weights_tensor, gamma=2.0)
"""
loss_fn = smp.losses.DiceLoss(
    mode='multiclass',
    smooth=1.0,        # tune if unstable
    log_loss=False,    # try True if training is slow to converge
    from_logits=True
)
"""
"""
loss_fn = smp.losses.TverskyLoss(
    mode='multiclass',      # 'binary', 'multiclass', or 'multilabel'
    alpha=0.3,              # weight of false positives
    beta=0.7,
    gamma= 2.0,
    smooth=1.0,
    log_loss=False,
    from_logits=True
)

"""


ce_loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
tversky_loss_fn = smp.losses.TverskyLoss(
    mode='multiclass',
    alpha=0.3, 
    beta=0.7,
    gamma=1.75,
    smooth=1.0,
    log_loss=False,
    from_logits=True
)

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, tv_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.tv_weight = tv_weight
        self.ce = ce_loss_fn
        self.tv = tversky_loss_fn

    def forward(self, logits, targets):
        """
        logits:      tensor of shape (B, C, H, W), raw model outputs
        targets:     tensor of shape (B, H, W), with class-indices in [0…C-1]
        """
        # Cross-entropy expects shape (B, C, H, W) + (B, H, W)
        ce_loss = self.ce(logits, targets)
        # print("ce_loss : {}".format(ce_loss))
        # Tversky also expects (B, C, H, W) + (B, H, W)
        tv_loss = self.tv(logits, targets)
        # print("tv_loss : {}".format(tv_loss))

        # combine
        loss = self.ce_weight * ce_loss + self.tv_weight * tv_loss
        return loss

# usage in your training loop
loss_fn = CombinedLoss(ce_weight=0.05, tv_weight=0.95)





# In[93]:

best_f1 = 0.0

for epoch in range(N_EPOCH):
    model.train()
    total_loss = 0

    # Wrap train_loader with tqdm to show progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCH}")

    for inputs, masks in progress_bar:

        inputs = inputs.to(DEVICE)
        masks = masks.long().to(DEVICE)
        outputs = model(inputs)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update the tqdm bar with loss
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(train_loader):.4f}")

       
    # evaluate
    avg_val_loss, _, weighted_f1_score = evaluate_pixel_based(model, val_loader, loss_fn, DEVICE)

    if weighted_f1_score >= best_f1 + 0.01:
        print(f"F1 improved ({best_f1:.4f} → {weighted_f1_score:.4f}). Saving model...")
        best_f1 = weighted_f1_score
        save_checkpoint(
            model, optimizer, epoch+1, avg_val_loss, BATCH_SIZE,
            filename=f"best_model_ce_tv_{MODEL_CODE}.pth"  
        )


    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(weighted_f1_score)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"Learning rate changed: {old_lr:.6f} → {new_lr:.6f}")




# In[ ]:


# Save after training
save_checkpoint(model, optimizer, epoch+1, loss.item(), BATCH_SIZE, filename="trained_model_{}.pth".format(MODEL_CODE))


# In[ ]:


# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights=None,
#     in_channels=3,
#     classes=1,
#     activation=None
# ).to(device)
# checkpoint = torch.load('trained_model.pth',map_location=torch.device("cpu"))
# model.load_state_dict(checkpoint['model_state_dict'])


# In[ ]:





def _object_level_metrics(pred_mask, true_mask, class_id=1, iou_threshold=0.5):
    """
    Compute TP, FP, FN for one image’s predicted vs. true mask for a given class.
    
    pred_mask, true_mask: 2D numpy arrays of shape (H, W), with integer class labels.
    class_id: the label of the object class you care about.
    """
    # Binary masks
    pred_bin = (pred_mask == class_id).astype(np.uint8)
    true_bin = (true_mask == class_id).astype(np.uint8)
    
    # Label connected components
    pred_labeled, num_pred = label(pred_bin)
    true_labeled, num_true = label(true_bin)
    
    matched_true = set()
    TP = 0
    
    # For each predicted object, find best‐matching true object
    for i in range(1, num_pred + 1):
        pred_obj = (pred_labeled == i)
        best_iou = 0
        best_j = None
        
        for j in range(1, num_true + 1):
            if j in matched_true:
                continue
            true_obj = (true_labeled == j)
            inter = np.logical_and(pred_obj, true_obj).sum()
            union = np.logical_or(pred_obj, true_obj).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_j = j
        
        if best_iou >= iou_threshold:
            TP += 1
            matched_true.add(best_j)
    
    FP = num_pred - TP
    FN = num_true - TP
    return TP, FP, FN

def evaluate_object_metrics(model, val_loader, device, class_id=1, iou_threshold=0.5):
    """
    Runs object‐based evaluation over the validation set and prints overall
    precision, recall, and F1 for the specified class_id.
    """
    model.eval()
    total_TP = total_FP = total_FN = 0
    
    with torch.no_grad():
        for inputs, masks in tqdm(val_loader, desc="Object-based evaluation"):
            inputs = inputs.to(device)
            outputs = model(inputs)                   # shape: (B, C, H, W)
            preds = outputs.argmax(dim=1).cpu().numpy()  
            trues = masks.long().cpu().numpy()          
            
            for p, t in zip(preds, trues):
                TP, FP, FN = _object_level_metrics(p, t, class_id, iou_threshold)
                total_TP += TP
                total_FP += FP
                total_FN += FN
    
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print(f"Object‐based Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1

for cls in range(NUM_CLASSES):
    print(f"Evaluating class {cls}")
    prec, rec, f1 = evaluate_object_metrics(model, val_loader, DEVICE, class_id=cls)


