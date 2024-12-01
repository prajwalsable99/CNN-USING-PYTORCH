# MNIST Handwritten Digit Classification using PyTorch

This repository contains a Convolutional Neural Network (CNN) implemented using PyTorch for classifying MNIST handwritten digits.

## Overview

This project demonstrates how to train a CNN model to classify MNIST digits, a dataset containing 28x28 grayscale images of handwritten digits from 0 to 9. The dataset is widely used for testing machine learning algorithms. The model was trained using the PyTorch framework.

### Key Components:
- **CNN Architecture**: The model uses two convolutional layers followed by two fully connected layers to classify the images.
- **Training**: The model is trained using the cross-entropy loss function and the Adam optimizer.
- **Evaluation**: The model's performance is evaluated using accuracy and loss metrics.

## Dataset

The dataset used in this project is the MNIST dataset, which is loaded using the `torchvision.datasets.MNIST` class. The dataset contains:
- **Training data**: 60,000 images of handwritten digits
- **Test data**: 10,000 images for testing

Each image is 28x28 pixels and is transformed into a tensor for input to the neural network.

## Model Architecture

The model is a CNN with the following layers:
1. **Conv1**: A convolutional layer with 10 filters, kernel size 5x5, padding 1.
2. **Conv2**: A convolutional layer with 20 filters, kernel size 5x5, padding 1.
3. **Fully Connected Layer 1**: A dense layer with 50 units.
4. **Output Layer**: A dense layer with 10 units, representing the 10 possible digits (0-9).

### Forward Pass:
- Input images are passed through the convolutional layers with ReLU activation and max pooling.
- The output is flattened and passed through fully connected layers.
- The final output is a set of probabilities for each digit class.

## Training and Results

The model was trained for 10 epochs. The following results were achieved:

| Epoch | Training Accuracy | Training Loss | Test Accuracy | Test Loss |
|-------|-------------------|---------------|---------------|-----------|
| 1     | 85%               | 0.53          | 95%           | 0.15      |
| 2     | 96%               | 0.12          | 98%           | 0.08      |
| 3     | 97%               | 0.08          | 98%           | 0.06      |
| 4     | 98%               | 0.07          | 98%           | 0.05      |
| 5     | 98%               | 0.06          | 99%           | 0.04      |
| 6     | 98%               | 0.05          | 99%           | 0.04      |
| 7     | 99%               | 0.04          | 99%           | 0.04      |
| 8     | 99%               | 0.04          | 99%           | 0.04      |
| 9     | 99%               | 0.04          | 99%           | 0.04      |
| 10    | 99%               | 0.03          | 99%           | 0.03      |

The model achieved a test accuracy of over 99% after 10 epochs.

## Requirements

To run the code, you will need the following libraries:
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `tqdm`

These can be installed via `pip`:

```bash
pip install torch torchvision numpy matplotlib tqdm
