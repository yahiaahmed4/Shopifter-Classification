# Shoplifting Detection Using Pre-trained ViViT Model

This repository contains the code for classifying shoplifting activities in videos using a fine-tuned ViViT (Video Vision Transformer) model. The notebook provided demonstrates the process of dataset preparation, model fine-tuning, evaluation, and inference on the shoplifting classification dataset. This project showcases how video classification can be applied to real-world scenarios like detecting shoplifting in surveillance footage.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Model](#model)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Inference](#inference)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Overview
The aim of this project is to classify videos into two categories: 
- **Non Shoplifters**
- **Shoplifters**

Using the **ViViT** model, a state-of-the-art transformer architecture for video understanding, this project fine-tunes the pre-trained model on a custom dataset of shoplifting activities. The repository contains the code for preparing the dataset, training the model, and evaluating its performance on unseen videos.

## Project Structure
- `Shop Dataset/`: Source folder for the raw video dataset containing two categories:
  - `non shop lifters/`
  - `shop lifters/`
- `ShopDataset/`: Target folder with the processed dataset structured into `train`, `val`, and `test` splits.
- `Fine-Tuning-ViVit-Shop-Lifter-Classification.ipynb`: Jupyter Notebook containing the entire code for dataset preparation, training, and evaluation.
- Pre-trained model link: [ViViT Model on Hugging Face](https://huggingface.co/google/vivit-b-16x2-kinetics400)

## Dataset Preparation
The dataset follows the structure of UCF-101, where videos are organized into train, validation, and test sets. The dataset is split into:
- **70% training**
- **15% validation**
- **15% testing**

Each video is processed, and statistics such as frame rates and durations are calculated to understand the dataset better.

Key dataset statistics:
- **Total videos**: 855
- **Longest video duration**: 74.00 seconds
- **Shortest video duration**: 3.00 seconds
- **Mean video duration**: 13.28 seconds
- **Mean frame rate**: 24.95 fps

## Model
The pre-trained ViViT model [google/vivit-b-16x2-kinetics400](https://huggingface.co/google/vivit-b-16x2-kinetics400) was used as the base model. Fine-tuning was done by replacing the classification head with a custom head for shoplifting classification.

The key components of the model:
- **ViViT architecture**: A transformer-based model designed for video classification tasks.
- **Pre-processing**: Videos are temporally subsampled to 32 frames, normalized, and resized.

## Training and Evaluation
### Training
The model was trained for **2 epochs** using a learning rate of `5e-5` and a batch size of `2`. The training utilized a **random clip sampler** to handle variable video lengths.

During training:
- **Loss**: 0.21
- **Training samples per second**: 0.133
- **Steps per second**: 0.066

### Evaluation
The evaluation was conducted on the validation and test sets. The model achieved:
- **Test Accuracy**: 100%
- **Validation Accuracy**: 100%

Evaluation statistics:
- **Eval Loss**: 0.0027
- **Eval Runtime**: 678.79 seconds
- **Eval Samples per Second**: 0.206

## Results
- **Training Loss**: 0.2103
- **Evaluation Loss**: 0.0027
- **Test Accuracy**: 100%

These results demonstrate the model's effectiveness in distinguishing between shoplifting and non-shoplifting behaviors in the given dataset.

## Inference
After fine-tuning the model, it is saved and can be loaded for inference on new videos. The inference process includes:
1. Loading a video and applying the pre-processing steps.
2. Running the model on the pre-processed video to predict the class (`shoplifter` or `non shoplifter`).

A pre-trained and fine-tuned model can be loaded from Hugging Face: [yehiawp4/ViViT-b-16x2-ShopLifting-Dataset](https://huggingface.co/yehiawp4/ViViT-b-16x2-ShopLifting-Dataset).

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yahiaahmed4/shoplifter-classification.git
   ```
2. Run the notebook to prepare the dataset, fine-tune the model, and evaluate it.

## Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- PytorchVideo
- OpenCV
- torchvision

## Acknowledgements
- The [ViViT model](https://huggingface.co/google/vivit-b-16x2-kinetics400) was utilized for video classification.
- Special thanks to the Hugging Face and PyTorch communities for providing excellent tools and resources.

