import os
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch_geometric.nn import GATConv
from torch_geometric.utils import grid
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- GAT Bottleneck (Improved for batched processing) ---
class GATBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        # GATConv expects [num_nodes, in_channels]
        self.gat1 = GATConv(in_channels, out_channels, heads=num_heads, concat=True, dropout=0.1)
        self.proj = nn.Linear(out_channels * num_heads, out_channels)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # x is now the batched node features: [B * N, C]
        x_in = x
        x = F.relu(self.gat1(x_in, edge_index))
        x = self.proj(x)
        x = self.dropout(x)

        # Add residual connection and normalization
        x = self.norm(x + x_in)
        return x

# --- GAT U-Net with ResNet50 Encoder ---
class GATUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # --- Encoder Layers from ResNet50 ---
        self.encoder1 = nn.Sequential(*list(base_model.children())[0:4]) # out: 64
        self.encoder2 = base_model.layer1 # out: 256
        self.encoder3 = base_model.layer2 # out: 512
        self.encoder4 = base_model.layer3 # out: 1024
        self.encoder5 = base_model.layer4 # out: 2048

        # --- Bottleneck ---
        self.proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.gat = GATBottleneck(256, 256)

        # --- Decoder Path ---
        self.upconv4 = nn.ConvTranspose2d(256, 1024, kernel_size=2, stride=2)
        self.conv4 = self._decoder_block(1024 + 1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv3 = self._decoder_block(512 + 512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv2 = self._decoder_block(256 + 256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = self._decoder_block(64 + 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def _align_and_cat(self, x_skip, x_up):
        """Interpolates upsampled tensor to match skip connection size."""
        return torch.cat([F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False), x_skip], dim=1)

    def forward(self, x):
        # --- Encoder Path ---
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # --- Bottleneck Path (with efficient batching) ---
        bottleneck = self.proj(x5)
        B, C, H, W = bottleneck.shape
        N = H * W

        # Reshape for GAT: [B, N, C] -> [B * N, C]
        graph_x_batched = bottleneck.view(B, C, N).permute(0, 2, 1).reshape(B * N, C)

        # Create batched edge_index
        edge_index_template, _ = grid(H, W, device=x.device)
        edge_indices = [edge_index_template + i * N for i in range(B)]
        edge_index_batched = torch.cat(edge_indices, dim=1)

        processed_graph_batched = self.gat(graph_x_batched, edge_index_batched)

        # Reshape back to feature map: [B * N, C] -> [B, N, C] -> [B, C, H, W]
        gat_map = processed_graph_batched.view(B, N, C).permute(0, 2, 1).view(B, C, H, W)

        # --- Decoder Path ---
        d4 = self.upconv4(gat_map)
        d4 = self.conv4(self._align_and_cat(x4, d4))
        d3 = self.upconv3(d4)
        d3 = self.conv3(self._align_and_cat(x3, d3))
        d2 = self.upconv2(d3)
        d2 = self.conv2(self._align_and_cat(x2, d2))
        d1 = self.upconv1(d2)
        d1 = self.conv1(self._align_and_cat(x1, d1))

        output = self.final(d1)
        return F.interpolate(output, size=(240, 320), mode='bilinear', align_corners=False)


# --- Dataset ---
class Fish4KDataset(Dataset):
    def __init__(self, img_dir, mask_dir, subfolders, transform=None):
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
        self.image_paths, self.challenges = [], []
        for sf in subfolders:
            folder = os.path.join(img_dir, sf)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(img_dir, sf, f))
                        self.challenges.append(sf)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.replace('imgs', 'masks').replace('test imgs', 'test masks').replace('.jpg', '.png')
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask, self.challenges[idx]

# --- Paths and DataLoaders (NO AUGMENTATIONS) ---
base = "D:/DAIICT/Sem 2/Minor Project/code/Minor Project"
subfolders = ["ComplexBkg", "Crowded", "DynamicBkg", "Hybrid", "Standard"]

base_transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Fish4KDataset(f"{base}/imgs", f"{base}/masks", subfolders, transform=base_transform)
test_dataset = Fish4KDataset(f"{base}/test imgs", f"{base}/test masks", subfolders, transform=base_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATUNet().to(device)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - ((2 * inter + smooth) / (union + smooth))

def combined_loss(pred, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    dice = dice_loss(pred, target)
    return bce + 2 * dice

# Automatically calculate pos_weight for class imbalance
print("Calculating pos_weight for BCE loss...")
num_positives, num_total = 0, 0
for _, masks, _ in tqdm(train_loader, desc="Calculating weights"):
    num_positives += torch.sum(masks == 1)
    num_total += masks.numel()
pos_weight = torch.tensor((num_total - num_positives) / (num_positives + 1e-6)).to(device)
print(f"Calculated pos_weight: {pos_weight.item():.2f}")

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

# --- Metrics ---
def compute_metrics(pred, target):
    pred_mask = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred_mask * target).sum()
    fp = pred_mask.sum() - tp
    fn = target.sum() - tp
    acc = ((pred_mask == target).float().sum() / target.numel()).item()
    precision = (tp / (tp + fp + 1e-6)).item()
    recall = (tp / (tp + fn + 1e-6)).item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
    return acc, precision, recall, f1

# --- Training Loop ---
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    total_loss, total_f1 = 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for imgs, masks, _ in progress_bar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = combined_loss(outputs, masks, pos_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, _, _, f1_score = compute_metrics(outputs.detach(), masks)
        total_f1 += f1_score
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", f1=f"{f1_score:.4f}")

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    avg_f1 = total_f1 / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}, F1-Score: {avg_f1:.4f}")

# --- Evaluation ---
print("\n--- Starting Evaluation ---")
model.eval()
metrics = {c: {"acc": [], "precision": [], "recall": [], "f1": []} for c in subfolders}

with torch.no_grad():
    for imgs, masks, chals in tqdm(test_loader, desc="Evaluating"):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        for i, chal in enumerate(chals):
            acc, prec, rec, f1 = compute_metrics(outputs[i:i+1], masks[i:i+1])
            metrics[chal]["acc"].append(acc)
            metrics[chal]["precision"].append(prec)
            metrics[chal]["recall"].append(rec)
            metrics[chal]["f1"].append(f1)

# --- Save Results and Visualize ---
final_metrics = []
for c in subfolders:
    final_metrics.append({
        "Challenge": c,
        "Testing Accuracy": f"{np.mean(metrics[c]['acc'])*100:.2f}",
        "Recall": f"{np.mean(metrics[c]['recall'])*100:.2f}",
        "Precision": f"{np.mean(metrics[c]['precision'])*100:.2f}",
        "F1-score": f"{np.mean(metrics[c]['f1'])*100:.2f}"
    })

df = pd.DataFrame(final_metrics)
print("\nFinal Metrics Table:")
print(df.to_string(index=False))

output_dir = "inference_output"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/final_metrics.csv", index=False)
torch.save(model.state_dict(), f"{output_dir}/gat_unet_model.pth")
print(f"\nModel and metrics saved successfully to '{output_dir}' folder.")

# Visualize a sample
with torch.no_grad():
    imgs, masks, _ = next(iter(test_loader))
    img_to_show = imgs[0].cpu().numpy().transpose(1, 2, 0)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_to_show = std * img_to_show + mean
    img_to_show = np.clip(img_to_show, 0, 1)

    outputs = model(imgs.to(device))
    pred_mask = (torch.sigmoid(outputs[0]) > 0.5).float().cpu().numpy().squeeze()
    ground_truth_mask = masks[0].cpu().numpy().squeeze()

    plt.figure(figsize=(15, 5))
    plt.subplot(131); plt.imshow(img_to_show); plt.title("Image"); plt.axis("off")
    plt.subplot(132); plt.imshow(ground_truth_mask, cmap="gray"); plt.title("Ground Truth Mask"); plt.axis("off")
    plt.subplot(133); plt.imshow(pred_mask, cmap="gray"); plt.title("Predicted Mask"); plt.axis("off")
    plt.savefig(f"{output_dir}/sample_plot.png")
    plt.show()