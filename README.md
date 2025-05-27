# DINOv2KAN: Kolmogorov-Arnold Network with DINOv2 Vision Transformer for Automatic Characterization of Alzheimer’s Disease

This repository contains the implementation and resources for **DINOv2-KAN**, a hybrid deep learning framework that combines the DINOv2 Vision Transformer (ViT) and Kolmogorov-Arnold Network (KAN) for MRI-based diagnosis of Alzheimer’s Disease (AD) with high accuracy.

## Overview

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that affects memory and cognition. Early and precise diagnosis is crucial for effective treatment, but traditional approaches often fail to capture subtle changes at different disease stages.  The proposed method addresses these challenges.

This hybrid model is trained with **supervised contrastive learning** to further improve classification accuracy. Extensive experiments on the ADNI, OASIS, and Kaggle datasets demonstrate its superior performance, achieving:
- **99.11% ± 0.25% accuracy** for Alzheimer’s Disease vs. Non-Cognitively Impaired classification.
- **99.37% ± 0.40% accuracy** for a four-class classification (AD, Non-Cognitively Impaired, Early Mild Cognitive Impairment, Mild Cognitive Impairment) on the ADNI dataset.

These results, validated on additional datasets, show statistically significant improvements over state-of-the-art methods.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Competing Models](#competing-models)
- [Proposed Model](#proposed-model)
- [Getting Started](#getting-started)
- [MRI Image Samples - Kaggle Dataset](#mri-image-samples---kaggle-dataset)


## Introduction

Alzheimer's disease (AD) is a progressive neurodegenerative disorder that affects millions of individuals worldwide. Early detection and accurate diagnosis are crucial for effective intervention. This work explores various deep learning models to enhance the accuracy of Alzheimer’s disease prediction using medical imaging data.

## Repository Structure
```md
Alzheimer-disease-prediction/
├── MODELS/
│   └── COMPETING MODELS/
│       ├── Alznet.ipynb
│       ├── HTLML.ipynb
│       ├── Modified Alexnet.ipynb
│       ├── Modified Inception.ipynb
│       ├── Resnet18.ipynb
│       ├── Resnet50.ipynb
│       ├── VGG16.ipynb
│       ├── VGG19.ipynb
│       └── ViTBiLSTM.ipynb
├── PROPOSED MODEL/
│   └── DINOV2KAN_Inference.ipynb
    └── DINOV2KAN_Train.ipynb
    └── Sample Dataset/
        └── ADNI
        └── Kaggle
        └── OASIS
├── Splits/
│   ├── kaggledataset_split.csv
│   ├── oasisdataset_split.csv
│   └── adnidataset_split.csv


```
### Competing Models

The `COMPETING MODELS` folder includes various well-known deep learning architectures that have been utilized to predict Alzheimer’s disease. Each model is implemented in a Jupyter Notebook and can be executed independently.

### Proposed Model

The `PROPOSED MODEL` folder features the **DINOv2-KAN** hybrid model, which combines vision transformers with Kolmogorov-Arnold Networks to enhance predictive performance. This notebook outlines the architecture, training process, and evaluation metrics.

## Getting Started

To get started with this repository:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Alzheimer-disease-prediction.git
   cd Alzheimer-disease-prediction

2. **Install the required dependencies:** 
   Make sure you have Python and pip installed, then run:
   ```bash
    pip install -r requirements.txt
3. **Run the notebooks:** 

### Step 1: Training

Open `PROPOSED MODEL/DINOV2KAN_Train.ipynb`
Set your dataset path:
   ```python
   # Choose your dataset
   dataset_choice = "Kaggle"  # Options: "ADNI", "Kaggle", "OASIS"
   dataset_path = f"Sample Dataset/{dataset_choice}/"
```
Execute all cells to start training.
The model checkpoint will be automatically saved at:
```python
/kaggle/working/
├── fold_1_best_model.pth
├── fold_2_best_model.pth
├── ...
```
These files represent the best checkpoint (based on accuracy) for each fold.
### Step 2: Testing/Inference
Open `PROPOSED MODEL/DINOV2KAN_Inference.ipynb`
Set your test dataset path and checkpoint path:
```python
# Use same dataset as training
dataset_choice = "Kaggle"  # Must match your training dataset
test_dataset_path = f"Sample Dataset/{dataset_choice}/"
```
Load the trained model checkpoint
```python
checkpoint_path = "/kaggle/working/fold_1_best_model.pth"
```
Execute all cells to run inference.

### Configuration Details
1. Model Architecture Parameters
```python
# DinoV2KAN Model Configuration
model_config = {
    'num_classes': 3,
    'dino_model': 'facebook/dinov2-base',  # Options: 'facebook/dinov2-base' or 'facebook/dinov2-large'
    'freeze_dino': True,
    'kan_config': {
        'dim': 768,               # DINOv2 feature dimension
        'num_heads': 12,
        'hdim_kan': 768,
        'mlp_ratio': 4.0,
        'drop': 0.1,
        'attn_drop': 0.1,
        'drop_path': 0.1
    }
}
```
2. Training Hyperparameters
```python
# Training Configuration
training_config = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 100,
    'patience': 15,                      # Early stopping patience
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
    'loss_function': 'CrossEntropyLoss'
}
```
3. Data Preprocessing Parameters
```python
# Preprocessing Configuration
preprocess_config = {
    'target_size': (224, 224),
    'skull_strip_threshold': 0.1,
    'resample_spacing': (1, 1, 1),
    'num_middle_slices': 50,
    'normalize_method': 'rescale_intensity',
    'clip_threshold': 0
}
```
## MRI Image Samples - Kaggle Dataset

<p>
  <img src="Images/MildDemented.jpg" alt="Caption 1" width="230"/>
  <img src="Images/ModerateDemented.jpg" alt="Caption 2" width="230"/>
  <img src="Images/NonDemented.jpg" alt="Caption 3" width="230"/>
  <img src="Images/VeryMildDemented.jpg" alt="Caption 4" width="230"/>
</p>

<p>
  <b>Mild Demented</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Moderate Demented</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Non Demented</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Very Mild Demented</b>
</p>







