Underwater Object Detection and Segmentation

📌 Overview

This project explores deep learning approaches for underwater object detection and segmentation.
Underwater environments pose unique challenges such as:

Poor visibility and lighting,

Dynamic and complex backgrounds,

Occlusions and overlapping objects.

We evaluate and compare three models:

U-Net

ResNet50

ResNet50 + Graph Attention Network (GAT)

The goal is to investigate how graph-based learning improves feature representation and segmentation performance in underwater imagery.

🧠 Methodology
1. Dataset

We use the Fish4Knowledge (F4K) dataset, which includes underwater scenes categorized into:

Complex Background (ComplexBkg)

Crowded

Dynamic Background (DynamicBkg)

Hybrid

Standard

Images are resized to 240×320 pixels with augmentations:

Random cropping, flipping, rotation, and color adjustments.

2. Models

U-Net: Encoder-decoder architecture for image segmentation.

ResNet50: Deep residual network modified with upsampling layers for segmentation.

ResNet50 + GAT: Graph Attention Network applied at the bottleneck to capture long-range dependencies in feature space.

3. Training Configuration

Optimizer: Adam, LR = 1e-4 with cosine annealing.

Loss: Binary Cross-Entropy + Dice loss.

Epochs: 200.

Gradient clipping, mixed precision training, and NaN checks for stability.

📊 Results

The figure below summarizes performance across challenges:

Key Observations

ResNet50 consistently achieves the best results, with highest F-measure and Precision across categories.

ResNet50+GAT performs closely, especially in Hybrid and Crowded scenarios, showing the benefit of attention-based context modeling.

U-Net lags behind due to its simpler architecture and lack of pretrained weights.

Dynamic backgrounds remain the most challenging scenario.

⚙️ Installation
# Clone repository
git clone https://github.com/aniruddha7599/underwater-object-detection.git
cd underwater-object-detection

# Create environment (optional but recommended)
conda create -n uw_detection python=3.9
conda activate uw_detection

# Install dependencies
pip install -r requirements.txt

🚀 Usage
Training
python train.py --model unet --dataset /path/to/F4K
python train.py --model resnet50 --dataset /path/to/F4K
python train.py --model resnet50_gat --dataset /path/to/F4K

Evaluation
python evaluate.py --model resnet50_gat --dataset /path/to/F4K --weights saved_models/resnet50_gat.pth

Inference
python inference.py --image sample.jpg --model resnet50_gat --weights saved_models/resnet50_gat.pth

📚 References

This project is based on a survey and experimental study presented in the minor project report:

M. Kapoor et al., Underwater Moving Object Detection using Encoder-Decoder Architecture and GraphSage, CVPR Workshop, 2023.

S. Gupta et al., DFTNet: Deep Fish Tracker With Attention Mechanism in Marine Environments, IEEE T-IM, 2021.

P. Sarkar et al., UICE-MIRNet guided image enhancement for underwater object detection, Scientific Reports, 2024.

O. Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI, 2015.

K. He et al., Deep Residual Learning for Image Recognition (ResNet), CVPR, 2016.

P. Veličković et al., Graph Attention Networks (GAT), ICLR, 2018.
