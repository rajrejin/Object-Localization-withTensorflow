# Emoji Detection and Bounding Box Prediction with TensorFlow

## Overview
This project demonstrates how to use **TensorFlow** to detect and classify emojis in an image, along with predicting their bounding boxes. The emojis are sourced from the **OpenMoji** dataset, and a custom neural network is trained to perform multi-task learning for both classification and regression tasks.

---

## Features
- **Emoji Dataset**: Downloads and processes emojis from the [OpenMoji project](https://openmoji.org).
- **Custom Data Generation**:
  - Generates synthetic training examples with random emoji placements.
  - Outputs include normalized images, one-hot encoded class labels, and bounding box coordinates.
- **Bounding Box Visualization**:
  - Ground truth and predicted bounding boxes are drawn on images for evaluation.
- **Multi-Task Neural Network**:
  - Predicts emoji classes and bounding box coordinates simultaneously.
  - Includes convolutional layers, batch normalization, and a fully connected head.
- **Custom IoU Metric**:
  - Implements Intersection Over Union (IoU) as a custom metric for evaluating bounding box predictions.
- **Callbacks**:
  - Visualizes test images during training.
  - Learning rate scheduler for dynamic adjustment.

---

## Key Libraries
- **TensorFlow**: For model building and training.
- **Matplotlib**: For visualization.
- **Pillow (PIL)**: For image processing.
- **NumPy**: For numerical operations.

---

## Setup

### Prerequisites
Install the following Python packages:
- `tensorflow`
- `matplotlib`
- `pillow`
- `numpy`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/emoji-detection.git
   cd emoji-detection
