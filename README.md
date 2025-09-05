# 🌊 Underwater Object Detection and Segmentation


---

## 📌 Overview
This project explores deep learning approaches for **underwater object detection and segmentation**.  
Underwater environments pose unique challenges such as:
- Poor visibility and lighting  
- Dynamic and complex backgrounds  
- Occlusions and overlapping objects  

We evaluate and compare three models:
- **U-Net**  
- **ResNet50**  
- **ResNet50 + Graph Attention Network (GAT)**  

The goal is to investigate how **graph-based learning** improves feature representation and segmentation performance in underwater imagery.

---

![Results](https://github.com/aniruddha7599/underwater-object-detection/blob/main/Model_comparison.png)


---

## 🧠 Methodology

### 1️⃣ Dataset
We use the **Fish4Knowledge (F4K)** dataset, which includes underwater scenes categorized into:
- Complex Background (ComplexBkg)  
- Crowded  
- Dynamic Background (DynamicBkg)  
- Hybrid  
- Standard  

**Preprocessing:**
- Images resized to **240×320 pixels**  
- Augmentations: random cropping, flipping, rotation, and color adjustments  

---

### 2️⃣ Models
- **U-Net** → Encoder-decoder architecture for segmentation.  
- **ResNet50** → Deep residual network with upsampling for pixel-level predictions.  
- **ResNet50 + GAT** → Adds a **Graph Attention Network** at the bottleneck to capture long-range dependencies.  

---

### 3️⃣ Training Configuration
- Optimizer: **Adam** (`lr = 1e-4`, cosine annealing scheduler)  
- Loss: **Binary Cross-Entropy + Dice Loss**  
- Epochs: **150**  
- Tricks: Gradient clipping, mixed precision training, NaN checks  

---

## 📊 Results

| Model         | ComplexBkg (F1) | Crowded (F1) | DynamicBkg (F1) | Hybrid (F1) | Standard (F1) |
|---------------|----------------|--------------|-----------------|-------------|---------------|
| **U-Net**     | 69.21          | 60.13        | 58.09           | 68.60       | 61.11         |
| **ResNet50**  | **83.91**      | **80.66**    | **71.54**       | **80.72**   | **78.47**     |
| **ResNet50+GAT** | 81.71       | 78.21        | 69.34           | 80.34       | 78.15         |

🔑 **Insights**:
- ResNet50 achieves the best overall results (highest **Precision** and **F-measure**).  
- ResNet50+GAT performs strongly in **crowded** and **hybrid** scenarios, showing the value of attention-based context modeling.  
- U-Net lags due to its simpler architecture and no pretrained backbone.  
- Dynamic backgrounds remain the toughest challenge.  

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/aniruddha7599/underwater-object-detection.git
cd underwater-object-detection

# Create environment (recommended)
conda create -n uw_detection python=3.9
conda activate uw_detection

# Install dependencies
pip install -r requirements.txt

```

📚 References

This work is based on a survey and experimental study presented in the minor project report:

M. Kapoor et al., Underwater Moving Object Detection using Encoder-Decoder Architecture and GraphSage, CVPR Workshop, 2023.

O. Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI, 2015.

K. He et al., Deep Residual Learning for Image Recognition (ResNet), CVPR, 2016.

P. Veličković et al., Graph Attention Networks (GAT), ICLR, 2018.
