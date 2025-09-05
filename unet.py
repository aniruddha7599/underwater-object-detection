import os
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Standard U-Net Architecture ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder (Contracting Path)
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)

        # Decoder (Expansive Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final convolution
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return F.interpolate(out, size=(240, 320), mode='bilinear', align_corners=False)

# --- Dataset ---
class Fish4KDataset(Dataset):
    def __init__(self, img_dir, mask_dir, subfolders, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.subfolders = subfolders
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
        self.image_paths = []
        self.challenges = []
        for sf in subfolders:
            folder = os.path.join(img_dir, sf)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(('.jpg', '.png')):
                        self.image_paths.append(os.path.join(sf, f))
                        self.challenges.append(sf)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_rel = self.image_paths[idx]
        img_path = os.path.join(self.img_dir, img_rel)
        mask_path = os.path.join(self.mask_dir, img_rel.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask, self.challenges[idx]

# --- Paths and DataLoaders ---
base = "C:/Users/Aniruddha shinde/Minor Project"
subfolders = ["ComplexBkg", "Crowded", "DynamicBkg", "Hybrid", "Standard"]
train_transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.RandomResizedCrop((240, 320), scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor()
])
train_dataset = Fish4KDataset(f"{base}/imgs", f"{base}/masks", subfolders, transform=train_transform)
test_dataset = Fish4KDataset(f"{base}/test imgs", f"{base}/test masks", subfolders, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum((2, 3))
    union = pred.sum((2, 3)) + target.sum((2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()

def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return bce + dice_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

# --- Gradient Clipping & Debugging ---
def clip_gradients(model, max_norm=0.5):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# --- Metrics ---
def compute_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum((2, 3))
    union = (pred + target).sum((2, 3)) - inter
    return (inter / (union + 1e-6)).mean().item()

def compute_accuracy(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    return ((pred == target).float().sum() / target.numel()).item()

def compute_precision(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    return ((pred * target).sum() / (pred.sum() + 1e-6)).item()

def compute_recall(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    return ((pred * target).sum() / (target.sum() + 1e-6)).item()

def compute_fmeasure(precision, recall):
    return 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) else 0.0

# --- Train ---
num_epochs = 150
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    model.train()
    total_loss, total_iou, total_acc = 0, 0, 0
    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        if torch.any(torch.isnan(imgs)) or torch.any(torch.isnan(masks)):
            print(f"NaN detected in input at epoch {epoch+1}")
            continue

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(imgs)
            loss = combined_loss(outputs, masks)
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}")
                break

        clip_gradients(model)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(outputs, masks)
        total_acc += compute_accuracy(outputs, masks)

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Acc: {avg_acc:.4f}")

# --- Evaluate and Visualize ---
model.eval()
metrics = {c: {"train_acc": [], "test_acc": [], "recall": [], "precision": []} for c in subfolders}

def collect_metrics(loader, split):
    for imgs, masks, chals in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        for i, chal in enumerate(chals):
            acc = compute_accuracy(outputs[i:i+1], masks[i:i+1])
            rec = compute_recall(outputs[i:i+1], masks[i:i+1])
            prec = compute_precision(outputs[i:i+1], masks[i:i+1])
            metrics[chal][f"{split}_acc"].append(acc)
            metrics[chal]["recall"].append(rec)
            metrics[chal]["precision"].append(prec)

with torch.no_grad():
    collect_metrics(train_loader, "train")
    collect_metrics(test_loader, "test")

    # Visualize a sample
    imgs, masks, chals = next(iter(test_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    outputs = model(imgs)
    pred_mask = (torch.sigmoid(outputs[0]) > 0.5).float().cpu().numpy().squeeze(0)
    ground_truth_mask = masks[0].cpu().numpy().squeeze(0)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(imgs[0].cpu().permute(1, 2, 0))
    plt.title("Image")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(ground_truth_mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.show()

for c in subfolders:
    for k in metrics[c]:
        metrics[c][k] = np.mean(metrics[c][k]) if metrics[c][k] else 0.0
    p, r = metrics[c]["precision"], metrics[c]["recall"]
    metrics[c]["fmeasure"] = compute_fmeasure(p, r)

# --- Save Results ---
df = pd.DataFrame({
    "Challenge": subfolders,
    "Training Accuracy": [f"{metrics[c]['train_acc']*100:.2f}" for c in subfolders],
    "Testing Accuracy": [f"{metrics[c]['test_acc']*100:.2f}" for c in subfolders],
    "Recall": [f"{metrics[c]['recall']*100:.2f}" for c in subfolders],
    "Precision": [f"{metrics[c]['precision']*100:.2f}" for c in subfolders],
    "F-measure": [f"{metrics[c]['fmeasure']*100:.2f}" for c in subfolders],
})
print("\nFinal Metrics Table:")
print(df.to_string(index=False))

os.makedirs("inference_output", exist_ok=True)
df.to_csv("inference_output/metrics.csv", index=False)
torch.save(model.state_dict(), "inference_output/unet_model.pth")
print("Model and metrics saved successfully.")