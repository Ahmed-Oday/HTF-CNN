# Multi-Task Emotion Recognition Model

## Description
This repository contains a multi-task deep learning model for emotion recognition that simultaneously performs:
- **7-class emotion classification** (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger)
- **Valence-Arousal regression** (continuous values in [0,1] range)

The model uses a VGG16 backbone with specialized branches for each task.

## Dataset Information
This project uses three datasets:
1. **CK+** (Extended Cohn-Kanade Dataset)
   - DOI/URL: https://doi.org/10.1109/CVPRW.2010.5543262
   - 593 sequences from 123 subjects
   - 7 emotion labels

2. **AffectNet**
   - DOI/URL: http://mohammadmahoor.com/affectnet/
   - ~450,000 images with 8 emotion labels
   - Valence-Arousal annotations

## Code Information
The repository contains:
- `multitask_model.py`: Multi-task VGG16 architecture with classification and regression branches
- `data_loader.py`: Custom dataset loaders for AffectNet with folder structure
- `train.py`: Main training script with early stopping and learning rate scheduling
- `evaluate.py`: Comprehensive evaluation with metrics and visualizations
- `config.py`: Configuration class with all hyperparameters
- `utils.py`: Utility functions for plotting and metrics calculation

## Requirements
```bash
# Core dependencies
python >= 3.8
pytorch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.19.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
tensorboard >= 2.7.0
pillow >= 8.3.0
opencv-python >= 4.5.0
tqdm >= 4.62.0
