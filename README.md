Deep Learning for Underwater Object Detection
This repository contains the official PyTorch implementation for the research paper "Deep Learning Approaches for Underwater Object Detection." It provides a benchmark and analysis of several deep learning architectures for segmenting objects in challenging underwater environments.

🎯 Overview
Object detection in underwater scenes is a significant challenge due to poor visibility, light scattering, complex backgrounds, and unpredictable object motion. This project implements and evaluates several deep learning models to tackle these problems, providing a robust framework for underwater image segmentation.

The core of this work is an empirical study of standard and graph-based neural network architectures on the Fish4Knowledge dataset.

✨ Key Features
Multiple Model Architectures: Implementation of U-Net, ResNet50, and a novel ResNet50+GAT model.

Graph Neural Network Integration: Explores the use of a Graph Attention Network (GAT) to model spatial dependencies between pixels for improved segmentation.

Comprehensive Evaluation: Models are benchmarked across five distinct and challenging underwater scenarios.

Modular Codebase: Easy-to-use scripts for training and evaluating models.

🏗️ Model Architectures
We implemented three distinct models to evaluate their effectiveness for underwater object segmentation.

U-Net: A classic fully convolutional neural network architecture for semantic segmentation. Its encoder-decoder structure with skip connections is highly effective at capturing contextual information and localizing objects precisely.

ResNet50: A powerful 50-layer deep residual network, pre-trained on ImageNet, serves as the backbone. We adapt it for segmentation by replacing the final fully connected layers with upsampling layers to generate a pixel-wise mask.

ResNet50 + GAT (Graph Attention Network): Our novel approach integrates a GAT module into the bottleneck of the ResNet50 architecture. This allows the model to learn the relative importance between different pixel regions, effectively modeling long-range dependencies and improving performance in cluttered or complex scenes.

📊 Dataset
The models were trained and evaluated on the Fish4Knowledge dataset. This dataset is specifically designed for underwater object detection and includes a wide variety of challenging scenarios.

The dataset is categorized into five distinct challenges:

ComplexBkg: Scenes with complex, non-uniform backgrounds.

Crowded: Scenes containing many overlapping objects.

DynamicBkg: Scenes with significant background motion.

Hybrid: A mixture of different challenges.

Standard: General underwater scenes with moderate difficulty.

Data Augmentation: To improve model generalization, we apply several augmentation techniques during training, including random cropping, horizontal flipping, rotation, and color jittering.

🚀 Results and Performance
The models were evaluated using Precision, Recall, and F-measure metrics. The ResNet50 architecture demonstrated the most robust and consistent performance across all categories, establishing a strong baseline. The ResNet50+GAT model also showed highly competitive results, confirming the potential of graph-based methods for this task.

Performance Metrics on Fish4Knowledge
Model

Challenge

F-measure (%)

Precision (%)

U-Net

Crowded

60.13

61.34

U-Net

DynamicBkg

58.09

59.96

ResNet50

ComplexBkg

83.91

85.62

ResNet50

Crowded

82.96

82.28

ResNet50+GAT

ComplexBkg

83.73

82.95

ResNet50+GAT

Hybrid

80.34

81.51

Visual Comparison Across Challenges
The following chart illustrates the performance of each model across the different challenge categories in the dataset.

🛠️ Getting Started
Follow these instructions to set up the environment and run the code.

Prerequisites
Python 3.8 or higher

PyTorch

A CUDA-enabled GPU is highly recommended for training.

Installation
Clone the repository:

git clone [https://github.com/your-username/underwater-object-detection.git](https://github.com/your-username/underwater-object-detection.git)
cd underwater-object-detection

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

(Note: You will need to create a requirements.txt file containing libraries like torch, torchvision, numpy, opencv-python, scikit-learn, etc.)

Training and Evaluation
Download the Dataset: Place the Fish4Knowledge dataset into a data/ directory within the project root.

Train a Model: Use the train.py script to start training.

python train.py --model ResNet50 --dataset_path ./data/ --epochs 200 --batch_size 8

Evaluate a Model: Use the evaluate.py script to test a trained model checkpoint.

python evaluate.py --model ResNet50 --weights_path ./checkpoints/resnet50_best.pth

🔮 Future Work
Based on our findings, future research will focus on:

Advanced GNNs: Exploring Hyperbolic Graph Convolutional Networks (HGCN) to better capture complex, non-Euclidean relationships in underwater scenes.

Image Enhancement: Integrating dedicated underwater image enhancement models (like UICE-MIRNet) as a preprocessing step to improve feature quality.

Object Tracking: Extending the current framework to include object tracking, implementing models reviewed in our paper like DFTNet.

📜 Citation
If you find this work useful in your research, please consider citing our paper and the original works that inspired this project.

@article{shinde2024deep,
  title={Deep Learning Approaches for Underwater Object Detection and Tracking: A Review},
  author={Shinde, Aniruddha Anil},
  year={2024},
  journal={Internal Report, Dhirubhai Ambani Institute of Information and Communication Technology}
}

This project builds upon the ideas from the following key papers:

U-Net: Convolutional Networks for Biomedical Image Segmentation

Deep Residual Learning for Image Recognition

Graph Attention Networks
