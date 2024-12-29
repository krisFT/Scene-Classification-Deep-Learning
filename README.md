# Scene Classification Using Deep Learning

This project implements a deep learning-based scene classification model capable of categorizing images into 15 distinct scene types. It includes:

1. A custom neural network designed from scratch.
2. A pre-trained VGG-16 model used for feature extraction, with only the classification head fine-tuned for this specific task.

---

## Features
- **Custom Neural Network Architecture**: Utilizes layers like `Conv2D`, `SeparableConv2D`, and `GlobalAveragePooling2D` for efficient feature extraction.
- **Pre-Trained VGG-16**: Leverages ImageNet weights for transfer learning, freezing the convolutional base and training only the classification head.
- **Data Augmentation**: Implemented in `preprocess.py`, with transformations including rotation, zoom, and flipping for improved generalization.
- **Dropout and Batch Normalization**: Reduces overfitting and stabilizes training.
- **Pre-Trained Weights and Logs Provided**: This repository includes:
  - Pre-trained VGG-16 weights (`vgg16_imagenet.h5`).
  - Best training weights for the custom model (`sc_model_checkpoints/weights.e48-acc0.7434.h5`) and the fine-tuned VGG model (`vgg_model_checkpoints/weights.e18-acc0.9049.h5`).
  - Training logs to facilitate evaluation and TensorBoard visualization.

---

## Training and Results

| Model                 | Total Parameters | Trainable Parameters | Epoch | Testing Accuracy  | Notes              |
|-----------------------|------------------|----------------------|-------|-------------------|--------------------|
| Custom Neural Network |    1,350,287     |       1,347,343      |  50   |**74.34%**         | Tuned for efficiency and simplicity. |
| VGG-16                |    21,172,815    |       6,457,871      |  20   |**90.49%**         | Fine-tuned classification head. |

- **Optimizer**: RMSprop with a learning rate of `1e-4` and momentum `0.09`.
- **Dataset**: Contains 15 scene categories with a total of 1,500 images.

---

## Custom Model Architecture

The custom model is designed to balance performance and parameter efficiency. Key highlights:
- **Convolutional and Depthwise Separable Layers**: Employs a mix of `Conv2D` and `SeparableConv2D` layers for computational efficiency and feature extraction.
- **Gradual Filter Expansion**: Filters increase progressively from 64 to 512 across blocks for hierarchical feature extraction.
- **Global Pooling**: Uses `GlobalAveragePooling2D` to reduce parameters compared to a fully connected layer after the convolutional stages.
- **Regularization**: Includes `Dropout` and L2 regularization to combat overfitting.

---

## System Setup for TensorFlow (on Windows 11)

Follow these steps to set up the environment on a Windows machine with an NVIDIA RTX 3050ti GPU:

1. **Create a Conda Environment:**
   ```bash
   conda create -n tf python=3.9
   conda activate tf
   ```
2. Install CUDA Toolkit and cuDNN:
   ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```
3. Install TensorFlow:
   ```bash
   pip install "tensorflow<2.11"
   ```
4. Downgrade NumPy:
   ```bash
   pip install "numpy<2.0"
   ```
5. **Download CUDA and cuDNN Files on Your Computer:**
   - [Download CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive): Ensure you select the appropriate version for your system.
   - [Download cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive): Match the cuDNN version with the installed CUDA version.
   - Copy cuDNN files to the corresponding CUDA folder on your system.

6. Install Additional Python Packages:
   ```bash
   pip install scikit-learn matplotlib pillow
   ```
### Notes:
- For detailed installation instructions, refer to the official [TensorFlow pip installation guide](https://www.tensorflow.org/install/pip#windows-native).
- Ensure compatibility between your TensorFlow, CUDA, and cuDNN versions by checking the [TensorFlow compatibility table](https://www.tensorflow.org/install/source_windows).

---

## How to Run

Follow the steps below to clone the repository, train the models, evaluate them, and monitor training using TensorBoard.

### 1. Clone the Repository
Start by cloning this repository to your local machine:
```bash
git clone https://github.com/krisFT/Scene-Classification-Deep-Learning.git
cd Scene-Classification-Deep-Learning
```

### 2. Train and Evaluate Models
#### Custom Neural Network (Task 1)
1. **Train the Model**:
```bash
python Scene_Classification_Deep_Learning.py --task 1 --data ./data
```
2. **Evaluate**:
```bash
python Scene_Classification_Deep_Learning.py --task 1 --data ./data --evaluate --load-checkpoint ./sc_model_checkpoints/weights.e48-acc0.7434.h5
```
#### VGG-16 Model (Task 2):
1. **Train the Model**:
```bash
python Scene_Classification_Deep_Learning.py --task 2 --data ./data
```
2. **Evaluate**:
```bash
python Scene_Classification_Deep_Learning.py --task 2 --data ./data --evaluate --load-checkpoint ./vgg_model_checkpoints/weights.e18-acc0.9049.h5
```

### 3. Monitor Training with TensorBoard
- Custom Model Logs:
  ```bash
  tensorboard --logdir logs/20241227-001937
  ```
- VGG Model Logs:
  ```bash
  tensorboard --logdir logs/20241229-164526
  ```




















