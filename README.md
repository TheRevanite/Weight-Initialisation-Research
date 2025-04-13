# Weight-Initialisation-Research


*Instructions to run all files*

# Initial Research
*1.2 Xavier-CNN-Cifar10.ipynb*
Description to Run the Code:
This script trains a custom 9-layer CNN on the CIFAR-10 dataset using PyTorch. It uses Xavier initialization, ReLU activations, and Adam optimizer. After 10 epochs of training, it reports the final test accuracy.

Steps:

1. Installs and loads CIFAR-10 with normalization.
2. Defines a CNN with 6 convolutional layers and 2 fully connected layers.
3. Applies Xavier uniform initialization.
4. Trains the model for 10 epochs.
5. Evaluates and prints test accuracy.
*To run: Just paste the code into a .py file or a Jupyter/Colab notebook and run it — it will train on GPU (if available).*
# Directory info-Initial Research/Xavier-CNN-Cifar10.py

*1.3 Xavier-Resnet-Cifar10.py*
Description to Run the Code:
This script trains a ResNet-18 model on the CIFAR-10 dataset using PyTorch. It replaces the final layer for 10-class classification and applies Xavier initialization to all Conv2d and Linear layers. The model is trained using ReLU activations and the Adam optimizer. After 10 epochs, it prints the final test accuracy.
Steps:
1. Loads and normalizes the CIFAR-10 dataset.
2. Initializes a ResNet-18 model with a modified final fully connected layer.
3. Applies Xavier uniform initialization to all Conv2d and Linear layers.
4. Trains the model for 10 epochs.
5. Evaluates and prints test accuracy.

*To run: Paste the code into a .py file or a Jupyter/Colab notebook and run it — it will train using GPU if available.*
# Directory info-Initial Research/Xavier-Resnet-Cifar10.py
# Initial Research
### Initial Trials/Poisson-Mish-CIFAR10.py:

This script trains a fully connected neural network on the **CIFAR-10** dataset using **custom Poisson-based weight initialization** and the **Mish activation function**. The model is implemented in PyTorch and evaluated over 10 epochs with classification metrics.

---

#### Steps:

1. Loads and normalizes the CIFAR-10 dataset.
2. Defines a **3-layer fully connected neural network** with:
   - Custom **Mish** activation function
   - Hidden layers: 256 → 128 → 10
3. Applies **Poisson-distributed weight initialization** to all layers with zero-centering, normalization, and a scaling factor.
4. Trains the network for **10 epochs** using **CrossEntropyLoss** and the **Adam optimizer**.
5. Evaluates the model on the test set and reports:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-score**

---

#### To Run:
Paste the code into a Python script or Jupyter/Colab notebook and run:

```bash
python Initial Trials/Poisson-Mish-CIFAR10.py


Here's a shorter version of the `README.md`:

---

# Custom CNN with Scaled Uniform Initialization

This repo trains a CNN on the CIFAR-10 dataset using a custom Scaled Uniform weight initializer in TensorFlow. The model is trained for 3 epochs with ReLU activation and the Adam optimizer.

## Steps to Run

1. **Install dependencies**:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

**3. Final Trials**
**3.1 Different-Datasets-Different-Models
3.1.1 CIFAR10
3.1.1.1 CNN_Cifar10.ipynb**

---

## 9-Layer CNN on CIFAR-10 with Hybrid Initialization

This repo implements a custom 9-layer CNN model trained on the CIFAR-10 dataset using PyTorch. The model uses **He initialization** for shallow layers and **Orthogonal initialization** for deeper layers to explore hybrid weight initialization strategies.

### Features

- Trains on CIFAR-10 using data augmentation
- Custom 9-layer CNN architecture
- Hybrid weight initialization: He (shallow) + Orthogonal (deep)
- Reports loss and accuracy every epoch

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

Install requirements:
```bash
pip install torch torchvision matplotlib
```

### How to Run

```bash
python CNN_Cifar10.ipynb
```

Make sure your script file is named `CNN_Cifar10.ipynb` or update the command accordingly.

---
**DIRECTORY INFO**-
```bash
Final Trials/Different-Datasets-Different-Models/CIFAR-10/CNN_Cifar10.ipynb
```
**3.1.1.2 CaffeNet_Cifar10.ipynb**


---

## CaffeNet on CIFAR-10 with Hybrid Initialization

This script implements a modified CaffeNet architecture trained on the CIFAR-10 dataset using PyTorch. A hybrid initialization strategy is applied: **He initialization** for the first 6 layers and **Orthogonal initialization** for the remaining ones.

### Features

- CaffeNet-style deep CNN with dropout regularization
- CIFAR-10 dataset with standard augmentations
- Custom hybrid weight initialization (He + Orthogonal)
- Cosine annealing learning rate scheduler
- Real-time training/validation plots

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

Install dependencies:
```bash
pip install torch torchvision matplotlib
```

### Run the Training

```bash
python CaffeNet_Cifar10.ipynb
```

**DIRECTORY INFO**-
```bash
Final Trials/Different-Datasets-Different-Models/CIFAR-10/CaffeNet_Cifar10.ipynb
```

**3.1.1.3 GoogleNet_Cifar10.ipynb**

---

## CIFAR-10 Classification using GoogLeNetSmall with Hybrid Initialization

This project implements a lightweight version of GoogLeNet (Inception-style architecture) for image classification on the CIFAR-10 dataset. It uses a custom hybrid weight initialization scheme: He initialization for the first few layers and Orthogonal initialization (with ReLU gain) for the rest.

---

### How to Run

1. Clone the repository or copy the script.
2. Install the required packages:
   ```bash
   pip install torch torchvision matplotlib
   ```
3. Run the script:
   ```bash
   GoogleNet_Cifar10.ipynb
   ```

The model will train for 50 epochs on CIFAR-10 using data augmentation and cosine annealing learning rate scheduling. Loss and accuracy plots will be displayed after training.

**DIRECTORY INFO-**
  ```bash
   Final Trials/Different-Datasets-Different-Models/CIFAR-10/GoogleNet_Cifar10.ipynb
   ```


**3.1.1.3 ResNet_Cifar10.ipynb**



---

## CIFAR-10 Classification using ResNet18 with Hybrid Initialization

This project implements a compact ResNet-style architecture for CIFAR-10 image classification. It uses a hybrid weight initialization strategy: He initialization for early layers and Orthogonal initialization (with ReLU gain) for deeper layers.

---

### How to Run

1. Clone the repository or copy the script.
2. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```
3. Run the script:
   ```bash
   ResNet_Cifar10.ipynb
   ```

The model trains on CIFAR-10 for 50 epochs using data augmentation and cosine annealing for learning rate scheduling. It outputs training/validation accuracy and loss curves along with total training time.

**DIRECTORY INFO-**
```bash
   Final Trials/Different-Datasets-Different-Models/CIFAR-10/ResNet_Cifar10.ipynb
   ```

---

**3.1.1.4 VGG_Cifar10.ipynb**
# CIFAR-10 Classification using VGGLike with Hybrid Initialization

This project implements a compact VGG-like convolutional neural network for CIFAR-10 image classification. It uses a hybrid weight initialization strategy: He initialization for the early layers and Orthogonal initialization (with ReLU gain) for the deeper layers.

## How to Run

Clone the repository or copy the script.

Install the required dependencies:

```bash
pip install torch torchvision matplotlib
```
Run The Script
```bash
python VGG_Cifar10.ipynb
```

**DIRECTORY INFO-**
```bash
   Final Trials/Different-Datasets-Different-Models/CIFAR-10/VGG_Cifar10.ipynb

   ```


**3.1.2.1-ResNet_Cifar100.ipynb**
# CIFAR-100 Classification using ResNetMini with Hybrid Initialization

This project implements a compact ResNet-like architecture for CIFAR-100 image classification. The model uses hybrid weight initialization: He initialization for the early layers and Orthogonal initialization (with ReLU gain) for deeper layers.

## Requirements

To run the code, you will need the following Python libraries:

- `torch`
- `torchvision`
- `matplotlib`

You can install the required dependencies with:

```bash
pip install torch torchvision matplotlib
```
How To Run
```bash
Python ResNet_Cifar100.ipynb
```

**DRIECTORY INFO-**
```bash
Final Trials/Different-Datasets-Different-Models/CIFAR-100/ResNet_Cifar100.ipynb
```

**3.1.2.2-VGG_Cifar100.ipynb**


---

# VGG-like Model on CIFAR-100 Dataset

This project implements a VGG-like CNN model for image classification on the CIFAR-100 dataset using PyTorch. It includes custom weight initialization and training for 50 epochs.

## Steps to Run the Code

1. **Install Dependencies**:
   Install the necessary libraries using pip:
   ```bash
   pip install torch torchvision matplotlib
   ```

2. **Run the Code**:
   Run the Python script to train and evaluate the model:
   ```bash
   python VGG_Cifar100.ipynb.py
   ```

3. **View Results**:
   After training, the training and validation accuracy/loss curves will be plotted. The time taken for training will also be printed in the console.


   **DIRECTORY INFO-**
      ```bash
   Final Trials/Different-Datasets-Different-Models/CIFAR-100/VGG_Cifar100.ipynb
   ```

---
**3.1.3-QMNIST**
**3.1.3.1-ORTH+HE-Caffenet-QMNIST.ipynb**

---

# CaffeNet on QMNIST Dataset

This project implements the CaffeNet model on the QMNIST dataset (32x32 grayscale images) using PyTorch. The model is trained for 10 epochs with custom weight initialization (He + Orthogonal) and evaluated on the test set.

## Steps to Run the Code

1. **Install Dependencies**:
   Install the necessary libraries using pip:
   ```bash
   pip install torch torchvision matplotlib
   ```

2. **Run the Code**:
   Run the Python script to train and evaluate the model:
   ```bash
   python ORTH+HE-Caffenet-QMNIST.ipynbpy
   ```

3. **View Results**:
   After training, the training and validation accuracy/loss curves will be plotted. The accuracy on the QMNIST test set will be printed in the console.
   **DIRECTORY INFO-**
   ```bash
   Final Trials/Different-Datasets-Different-Models/QMNIST/ORTH+HE-Caffenet-QMNIST.ipynb
   ```

---

**3.1.3.2-ORTH+HE_CNN_QMNIST.ipynb**


---

# QMNIST Classification using Custom 9-Layer CNN

This project uses a custom 9-layer Convolutional Neural Network to classify images from the QMNIST dataset. It applies a hybrid weight initialization strategy: He initialization for the shallow layers and Orthogonal initialization for the deeper ones. The model is trained using CrossEntropyLoss and Adam optimizer, with Cosine Annealing learning rate scheduling and L2 regularization.

---

## Requirements

- Python 3.x  
- PyTorch  
- torchvision  
- matplotlib  

You can install dependencies with:

```bash
pip install torch torchvision matplotlib
```

---

## Running the Code

1. Save the script in a Python file (e.g., `ORTH+HE_CNN_QMNIST.ipynb`).
2. Run the script:

```bash
python ORTH+HE_CNN_QMNIST.ipynb
```

The script will:
- Download and preprocess the QMNIST dataset
- Initialize the CNN with custom weight initialization
- Train and validate the model for 50 epochs
- Plot training/validation loss and accuracy

---

## Notes

- The model uses GPU (CUDA) if available.
- QMNIST images are resized to 32x32 before training.
- Training progress and plots will appear after training.


  **DIRECTORY INFO-**
 ```bash
  Final Trials/Different-Datasets-Different-Models/QMNIST/ORTH+HE_CNN_QMNIST.ipynb
```
---


**3.1.3.3 Final Trials/Different-Datasets-Different-Models/QMNIST/ORTH+HE_googlenet_QMNIST.ipynb**



---

# QMNIST Classification with GoogleNet

This project uses the GoogleNet architecture to classify handwritten digits from the QMNIST dataset. The grayscale images are resized to 32x32 and converted to 3-channel RGB to be compatible with GoogleNet. The model applies a custom weight initialization strategy: He initialization for convolutional layers and Orthogonal initialization for linear layers.

---

## Requirements

- Python 3.x  
- PyTorch  
- torchvision  
- matplotlib  

Install dependencies with:

```bash
pip install torch torchvision matplotlib
```

---

## How to Run

1. Save the code in a Python file (e.g., `qmnist_googlenet.py`).
2. Run the script:

```bash
python qmnist_googlenet.py
```

This will:

- Download and preprocess the QMNIST dataset
- Apply custom initialization to GoogleNet
- Train for 10 epochs and validate on a held-out validation set
- Print final test accuracy
- Plot training and validation accuracy over epochs

---

## Notes

- Uses GPU if available.
- Converts QMNIST grayscale images to RGB format.
- Batch size: 64  
- Optimizer: Adam  
- Loss function: CrossEntropyLoss  
- Dataset is split 90% training, 10% validation.  

---

Let me know if you want this combined with the other readmes into a single multi-model benchmark doc.

  **DIRECTORY INFO-**
 ```bash
  Final Trials/Different-Datasets-Different-Models/QMNIST/ORTH+HE_googlenet_QMNIST.ipynb
```
---





**3.1.3.3 Final Trials/Different-Datasets-Different-Models/QMNIST/orth+he_resnet_qmnist.ipynb**




# QMNIST Classification with ResNet18 (Custom Initialization)

This project implements digit classification on the QMNIST dataset using a modified ResNet18 architecture in PyTorch. It includes a custom weight initialization strategy that combines **He initialization** for convolutional layers and **Orthogonal initialization** for fully connected layers.

## Features

- Uses the **QMNIST** dataset with 32x32 grayscale images.
- Implements a modified **ResNet18** to handle single-channel inputs.
- Applies **custom hybrid initialization** (He + Orthogonal).
- Trains with **Adam optimizer** and **CrossEntropy loss**.
- Includes validation and test evaluation.
- Plots training and validation accuracy/loss curves.

## Steps to Run

1. **Install dependencies** (PyTorch, torchvision, matplotlib):

   ```bash
   pip install torch torchvision matplotlib
   ```

2. **Run the script**:

   ```bash
   python orth+he_resnet_qmnist.ipynb.py
   ```

3. The script will:
   - Download the QMNIST dataset.
   - Train ResNet18 for 10 epochs with custom initialization.
   - Print accuracy metrics.
   - Display plots for training/validation accuracy and loss.



## Notes

- The model is trained on GPU if available, else CPU.
- No pre-trained weights are used.
- Custom initialization is applied using `kaiming_normal_` and `orthogonal_`.

---


  **DIRECTORY INFO-**
 ```bash
  Final Trials/Different-Datasets-Different-Models/QMNIST/orth+he_resnet_qmnist.ipynb
```
---



**3.1.4 STL10**

**3.1.4.1 ORTH+HE-resnet-STL10.ipynb**
Sure, here's the updated README without the output example:

---

# STL10 Image Classification with Modified ResNet18

This project implements image classification on the STL10 dataset using a modified ResNet18 architecture in PyTorch. It introduces custom weight initialization combining He and Orthogonal methods, includes dropout regularization, and uses Cosine Annealing for learning rate scheduling. The model is trained with data augmentation and evaluated on classification accuracy.

## Features

- Uses ResNet18 with a custom final fully connected layer and dropout  
- Custom weight initialization: He for shallow layers, Orthogonal for deep layers  
- Data augmentation with color jittering, random crops, and horizontal flips  
- Cosine Annealing Learning Rate Scheduler  
- Training and validation accuracy/loss tracking with visualization  

## Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- matplotlib  

You can install dependencies via:

```bash
pip install torch torchvision matplotlib
```

## Steps to Run the Code

1. **Download the Dataset**  
   The STL10 dataset will be automatically downloaded the first time you run the code.

2. **Train the Model**  
   Run the script to start training and evaluation:
   ```bash
   python ORTH+HE-resnet-STL10.ipynb
   ```

3. **Visualize Results**  
   After training, loss and accuracy curves will be plotted automatically for both training and validation sets.

## Notes

- Resize is applied to downsample STL10 images from 96x96 to 64x64 for compatibility.  
- Training is performed on GPU if available.  
- Dropout (p=0.5) is used in the final classifier to reduce overfitting.  

---





  **DIRECTORY INFO-**
 ```bash
  Final Trials/Different-Datasets-Different-Models/QMNIST/ORTH+HE-resnet-STL10.ipynb
```
---

**3.1.4.2 ORTH+HE_caffenet_STL10 (1).ipynb**




---

# STL10 Image Classification with AlexNet

This project performs image classification on the STL10 dataset using a modified AlexNet (CaffeNet) architecture in PyTorch. It includes custom weight initialization combining He and Orthogonal methods, and applies standard data preprocessing and augmentation techniques.

## Features

- AlexNet pretrained on ImageNet, adapted for STL10 (10 classes)  
- Custom weight initialization: He for convolutional layers, Orthogonal for fully connected layers  
- Input resizing to 224×224 to match AlexNet's input requirement  
- Accuracy and loss tracking for both training and validation phases  
- Visualization of learning curves (accuracy and loss)  

## Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- matplotlib  

Install dependencies:

```bash
pip install torch torchvision matplotlib
```

## Steps to Run the Code

1. **Clone the Repository or Copy the Script**  
   Save the provided Python script to your local machine.

2. **Run the Script**  
   Execute the script to train and evaluate the model:
   ```bash
   python ORTH+HE_caffenet_STL10 (1).ipynb
   ```

3. **Visualize Learning Curves**  
   Accuracy and loss over epochs will be automatically plotted after training.

## Notes

- STL10 dataset will be automatically downloaded the first time the script is run.  
- Training is done using GPU if available.  
- The final fully connected layer is modified to classify 10 STL10 categories.  

---



  **DIRECTORY INFO-**
 ```bash
  Final Trials/Different-Datasets-Different-Models/QMNIST/ORTH+HE_caffenet_STL10 (1).ipynb
```
---
**3.1.5 SVHN**
**3.1.5.1 CNN_SVHN.ipynb*


---

# SVHN Classification using 9-Layer CNN

This project implements a 9-layer Convolutional Neural Network (CNN) to classify images from the **Street View House Numbers (SVHN)** dataset using PyTorch. SVHN is a real-world image dataset for developing machine learning and object recognition algorithms, obtained from house numbers in Google Street View images.

## Features

- SVHN dataset loading (from `.mat` files or torchvision API)
- Custom 9-layer CNN architecture
- Training with Adam optimizer and cross-entropy loss
- Evaluation on test data to compute accuracy

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scipy (if using `.mat` files directly)

Install required libraries with:

```bash
pip install torch torchvision
```

## Steps to Run

1. **Save the code in a Python file** (e.g., `cnn_svhn.py`).

2. **Run the script**:

```bash
python cnn_svhn.py
```

3. The SVHN dataset will be downloaded automatically. The model will train for 10 epochs and display the training loss after each epoch and final test accuracy.

## Model Architecture

The 9-layer CNN consists of:

- Three convolutional blocks:
  - Each with 2 or more Conv2D + ReLU layers
  - Followed by MaxPooling
- Fully connected classifier with:
  - Linear layers
  - ReLU activation
  - Dropout
- Output layer with 10 classes (digits 0–9)

## Dataset Info

SVHN contains over 600,000 digit images (32x32 RGB), split into training and test sets. This script uses the torchvision loader to fetch and normalize the data automatically.

---


  **DIRECTORY INFO-**
 ```bash
  Final Trials/Different-Datasets-Different-Models/SVHN/CNN_SVHN.ipynb
```
---
**3.1.5.2 ResNet_SVHN.ipynb**
# SVHN Classification using ResNet Mini

This project implements a ResNet Mini architecture for classifying images from the **SVHN (Street View House Numbers)** dataset. The model is built using PyTorch and uses a custom initialization strategy with Kaiming and Orthogonal initializations. The script trains the model for 50 epochs, evaluates it on the test set, and plots training/validation loss and accuracy.

## Features

- Uses the **SVHN** dataset (downloaded automatically via torchvision).
- Implements a **ResNet Mini** model with residual blocks.
- Custom weight initialization using Kaiming and Orthogonal methods.
- Uses Adam optimizer and Cosine Annealing learning rate scheduler.
- Tracks training and validation loss/accuracy.
- Visualizes the training process with loss and accuracy curves.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib

Install required libraries with:

```bash
pip install torch torchvision matplotlib
```

## Steps to Run

1. **Save the code** in a Python file (e.g., `ResNet_SVHN.ipynb`).

2. **Run the script** using:

```bash
python ResNet_SVHN.ipynb
```

3. The SVHN dataset will be automatically downloaded. The model will train for 50 epochs, and both training and validation loss/accuracy will be plotted at the end.

## Model Architecture

- **ResNet Mini** consists of:
  - Three residual blocks with 2 layers each.
  - Adaptive Average Pooling followed by a fully connected layer.
- Custom initialization applied to convolutional and fully connected layers.

## Dataset Info

SVHN is a real-world image dataset containing over 600,000 labeled digits (32x32 RGB) obtained from street view images. This script uses the torchvision loader to fetch and normalize the data automatically.

---

  **DIRECTORY INFO-**
 ```bash
  Final Trials/Different-Datasets-Different-Models/SVHN/ResNet_SVHN.ipynb
```
---



**3.1.5.3 VGG_SVHN.ipynb**



## Description

This repository contains code for training a VGG-like convolutional neural network (CNN) on the SVHN dataset. The model uses a custom weight initialization technique that combines Kaiming Normal initialization for the first few layers and Orthogonal initialization for the deeper layers. The training process utilizes the Adam optimizer and Cosine Annealing Learning Rate Scheduler. The model is evaluated on both training and validation accuracy and loss.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib

You can install the required libraries using pip:

```bash
pip install torch torchvision matplotlib
```

## Dataset

The model is trained on the **SVHN (Street View House Numbers)** dataset, which is a large collection of labeled digits from real-world images.

The dataset is automatically downloaded when running the code.

## Model

The VGG-like model consists of:
- Several convolutional layers followed by ReLU activations and Batch Normalization.
- Max-Pooling layers to down-sample the feature maps.
- Fully connected layers at the end for classification.

A custom initialization function is applied to the model:
- The first 6 layers use Kaiming Normal initialization.
- Subsequent layers use Orthogonal initialization.

## Steps to Run the Code

1. **Prepare the environment** by installing the required dependencies:

   ```bash
   pip install torch torchvision matplotlib
   ```

2. **Run the script**:

   ```bash
   python vgg_svh.py
   ```

3. The code will:
   - Download the SVHN dataset.
   - Initialize the model with custom weights.
   - Train the model for 50 epochs using the Adam optimizer.
   - Plot the training/validation loss and accuracy curves after training.

## Model Training and Evaluation

The model is trained for 50 epochs and uses the following hyperparameters:

- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Learning Rate Scheduler**: Cosine Annealing

**DIRECTORY INFO-**
 ```bash
Final Trials/Different-Datasets-Different-Models/SVHN/VGG_SVHN.ipynb
```
---











