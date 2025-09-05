import os
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- U-Net with ResNet Encoder ---
class UNetResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        self.proj = nn.Conv2d(2048, 256, kernel_size=1)

        self.upconv4 = nn.ConvTranspose2d(256, 512, 2, 2)
        self.conv4 = nn.Conv2d(1536, 512, 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv3 = nn.Conv2d(768, 256, 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 64, 2, 2)
        self.conv2 = nn.Conv2d(320, 64, 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.final = nn.Conv2d(32, num_classes, 1)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x1 = self.encoder[0:4](x)
        x2 = self.encoder[4](x1)
        x3 = self.encoder[5](x2)
        x4 = self.encoder[6](x3)
        x5 = self.encoder[7](x4)

        x5 = self.proj(x5)

        x = self.upconv4(x5)
        x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv4(torch.cat([x, x4], dim=1))

        x = self.upconv3(x)
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv3(torch.cat([x, x3], dim=1))

        x = self.upconv2(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.upconv1(x)
        x = self.bn(x)
        x = self.final(x)

        return F.interpolate(x, size=(240, 320), mode='bilinear', align_corners=False)

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
base = "D:\DAIICT\Sem 2\Minor Project\code\Minor Project"
subfolders = ["ComplexBkg", "Crowded", "DynamicBkg", "Hybrid", "Standard"]

# Define a single, non-augmenting transform for both training and testing
base_transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor()
])

# Apply the same base_transform to both datasets
train_dataset = Fish4KDataset(f"{base}/imgs", f"{base}/masks", subfolders, transform=base_transform)
test_dataset = Fish4KDataset(f"{base}/test imgs", f"{base}/test masks", subfolders, transform=base_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)


# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetResNet50().to(device)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum((2, 3))
    union = pred.sum((2, 3)) + target.sum((2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()

def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return bce + dice_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

# --- Gradient Clipping & Debugging ---
def clip_gradients(model, max_norm=0.5):  # Stronger clipping
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
num_epochs = 150  # Increased epochs
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    model.train()
    total_loss, total_iou, total_acc = 0, 0, 0
    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        # Check for NaNs
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
    pred_mask = (torch.sigmoid(outputs[0]) > 0.5).float().cpu().numpy().squeeze(0)  # Remove channel dim
    ground_truth_mask = masks[0].cpu().numpy().squeeze(0)  # Remove channel dim

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
torch.save(model.state_dict(), "inference_output/unet_resnet_model.pth")
print("Model and metrics saved successfully.")