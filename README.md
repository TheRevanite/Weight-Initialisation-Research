# Weight-Initialisation-Research


*Instructions to run all files*

# Initial Research

**1.1 LSUV.ipynb**


---

# CIFAR-10 Classification using 9-Layer CNN with LSUV Initialization

This project implements a 9-layer Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch. It uses **LSUV (Layer-sequential unit-variance)** initialization combined with orthogonal weight initialization to stabilize training.

## Features

- Custom 9-layer CNN architecture
- LSUV initialization for better convergence
- Batch Normalization and Dropout for regularization
- Cosine Annealing Learning Rate Scheduler
- Accuracy and loss visualization

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib

Install dependencies using:

```bash
pip install torch torchvision matplotlib
```

## Dataset

The CIFAR-10 dataset will be automatically downloaded via `torchvision.datasets`.

## Steps to Run

1. Clone this repository or copy the script into a `.py` file.
2. Make sure your environment supports CUDA (optional for GPU acceleration).
3. Run the training script:

```bash
python LSUV.ipynb
```

4. After training, loss and accuracy graphs for both training and validation sets will be displayed.

## Notes

- LSUV initialization uses a single batch from the training data to normalize activations to unit variance.
- Training is performed for 50 epochs using Adam optimizer and cosine annealing scheduling.
- Batch size is set to 64.

--- 
**DIRECTORY INFO-**
```bash
Initial Research/LSUV.ipynb
```
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


**3.2 ORTH+HE-CNN-CIFAR100**
**3.2.1 cifar100(orth+he).ipynb**



## Description

This repository contains code to train a custom convolutional neural network (CNN) on the CIFAR-100 dataset. The network is composed of multiple convolutional layers followed by fully connected layers. The custom CNN model uses orthogonal initialization for the weights and is optimized using the Adam optimizer with a cross-entropy loss function. The code trains the model and evaluates its accuracy on the CIFAR-100 test set.

## Requirements

- Python 3.x
- PyTorch
- torchvision

You can install the required libraries using pip:

```bash
pip install torch torchvision
```

## Dataset

The model is trained on the **CIFAR-100** dataset, which consists of 100 different classes of images with 60,000 32x32 color images. The dataset is automatically downloaded when running the code.

## Model

The custom CNN model architecture consists of:
- Several convolutional layers followed by ReLU activations.
- Max-pooling layers for down-sampling.
- An adaptive average pooling layer.
- Fully connected layers for classification into 100 classes.

A custom weight initialization function is used:
- The weights of convolutional and fully connected layers are initialized using orthogonal initialization.

## Steps to Run the Code

1. **Prepare the environment** by installing the required dependencies:

   ```bash
   pip install torch torchvision
   ```

2. **Run the script**:

   ```bash
   python custom_cnn_cifar100.py
   ```

3. The code will:
   - Download the CIFAR-100 dataset.
   - Initialize the model with custom weights.
   - Train the model for 10 epochs using the Adam optimizer.
   - Display the loss and accuracy for each epoch.

## Model Training and Evaluation

The model is trained for 10 epochs with the following hyperparameters:

- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss

The accuracy of the model is evaluated after each epoch on the test dataset.

**DIRECTORY INFO-**
```bash
Final Trials/ORTH+HE-CNN-CIFAR100/cifar100(orth+he).ipynb
```
---
**3.3 ORTH+HE-CNN-QMNIST**
**3.3.1 ORTH+HE_CNN_QMNIST.ipynb**





---

# QMNIST Classification using Custom 9-Layer CNN

This project implements a deep convolutional neural network (CNN) to classify grayscale handwritten digits from the QMNIST dataset using a hybrid weight initialization strategy: **He initialization** for shallow layers and **Orthogonal initialization** for deeper layers.

## Features

- Dataset: QMNIST (extended MNIST)
- Architecture: 9-layer custom CNN with dynamic shape handling
- Hybrid Initialization: He + Orthogonal
- Regularization: L2 (weight decay)
- Learning Rate Scheduler: Cosine Annealing
- Evaluation: Training and validation accuracy/loss tracking
- Visualization: Epoch-wise training/validation loss and accuracy plots

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

Install dependencies using:

```bash
pip install torch torchvision matplotlib
```

---

## Steps to Run

1. **Clone the Repository** or copy the code into a `.py` file (e.g., `ORTH+HE_CNN_QMNIST.ipynb`).

2. **Run the script**:

```bash
python ORTH+HE_CNN_QMNIST.ipynb
```

3. The script will:
   - Download and preprocess the QMNIST dataset.
   - Train the model for 50 epochs.
   - Track and print training/validation accuracy and loss.
   - Display performance graphs after training.

---

## Notes

- Model uses grayscale images resized to 32x32.
- L2 regularization (weight decay) is applied to reduce overfitting.
- Learning rate is annealed smoothly over epochs using a cosine scheduler.

---

Let me know if you'd like a version in `.md` format or if you're planning to upload this to GitHub and need additional sections like license, contribution, or citation.

**DIRECTORY INFO-**
```bash
Final Trials/ORTH+HE-CNN-QMNIST/ORTH+HE_CNN_QMNIST.ipynb
```
---
**3.4 ORTH+HE-CNN-STL10**
**3.4.1 stl10(orth+he).ipynb**




---

# STL-10 Image Classification using CNN

This project implements a deep Convolutional Neural Network (CNN) using PyTorch to classify images from the [STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/). It features custom weight initialization, data augmentation, validation tracking, test accuracy evaluation, learning rate scheduling, and early stopping.

## Model Overview

- 3 Convolutional blocks (64 → 128 → 256 channels)
- Batch Normalization & ReLU activations
- MaxPooling and Dropout for regularization
- Fully Connected Classifier with Dropout
- Custom weight initialization (He/Kaiming Normal)

## Dataset

- **STL-10**: Contains 10 classes of color images (96x96), suitable for unsupervised feature learning and image classification.
- Training split: 80% training, 20% validation
- Preprocessing includes resize, normalization, and data augmentation (horizontal flip)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib

Install dependencies using pip:

```bash
pip install torch torchvision matplotlib
```

## How to Run

1. **Clone the repository** or copy the code.
2. **Run the script** directly to start training:

```bash
python your_script_name.py
```

3. The model will:
   - Train with early stopping
   - Save the best model to `best_model.pt`
   - Plot training loss, validation accuracy, and test accuracy

## Notes

- Training uses the GPU if available (`torch.cuda.is_available()`).
- Uses Adam optimizer with learning rate decay and weight decay for regularization.
- The model architecture and training strategy are optimized specifically for the STL-10 dataset.

---




**DIRECTORY INFO-**

```bash
Final Trials/ORTH+HE-CNN-STL10/stl10(orth+he).ipynb
```
---


**3.5 SVHN**
**3.5.1 Final Trials/SVHN/svhn(orth+he).ipynb**


---

# SVHN Classification using Custom 9-Layer CNN

This project implements a deep 9-layer Convolutional Neural Network (CNN) to classify images from the SVHN (Street View House Numbers) dataset. The model is trained using PyTorch with Orthogonal weight initialization scaled by √2, optimized with Adam.

## Dataset

- **SVHN (Street View House Numbers)**  
  Downloaded automatically via `torchvision.datasets.SVHN`.

## Model Architecture

- 9-layer deep CNN with:
  - Three convolutional blocks
  - ReLU activations
  - MaxPooling and AdaptiveAvgPooling
  - Fully connected layers
- Custom weight initialization using Orthogonal + √2 scaling

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- matplotlib (optional, if you plan to add visualizations)

Install dependencies using:

```bash
pip install torch torchvision
```

## Steps to Run

1. Clone this repository or copy the code into a Python script.
2. Run the script to:
   - Automatically download and preprocess the SVHN dataset.
   - Train the custom CNN model on the training split.
   - Evaluate the model on the test split after each epoch.

```bash
pythonsvhn(orth+he).ipynb**
```

## Notes

- The model uses GPU (CUDA) if available.
- Batch size: 128
- Optimizer: Adam with learning rate 0.001
- Loss Function: CrossEntropyLoss
- Number of Epochs: 50

---

**DIRECTORY INFO-**
```bash
Final Trials/SVHN/svhn(orth+he).ipynb
```
---

**4 FINAL**
**4.1 cifar10(orth+He)-Resnet.py**

:

---

# CIFAR-10 Classification using Custom ResNet with Hybrid Initialization

This project implements a simplified ResNet-like convolutional neural network to classify images from the CIFAR-10 dataset using PyTorch. It includes a hybrid initialization scheme using He (Kaiming) initialization for early layers and Orthogonal initialization for deeper layers.

## Features

- Custom ResNetMini architecture with residual blocks
- CIFAR-10 dataset with standard data augmentations
- Cosine annealing learning rate scheduler
- Hybrid initialization strategy (He + Orthogonal)
- Accuracy and loss tracking with visual plots

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib

You can install the dependencies using:

```bash
pip install torch torchvision matplotlib
```

## Steps to Run

1. Clone the repository or copy the script.

2. Run the training script:

```bash
python cifar10(orth+He)-Resnet.py
```

3. The script will automatically:
   - Download the CIFAR-10 dataset
   - Train the custom ResNet model
   - Display training and validation curves for accuracy and loss

## Notes

- Model is trained for 50 epochs using Adam optimizer.
- Data augmentation includes random crop and horizontal flip.
- Results are visualized at the end using matplotlib.

--- 


**DIRECTORY INFO-**
```bash
Final/cifar10(orth+He)-Resnet.py
```

**4.2 cifar10(orth+He)-VGG.py**




---

# VGG-like CNN on CIFAR-10

This project trains a custom VGG-style convolutional neural network on the CIFAR-10 dataset using PyTorch. The model utilizes hybrid weight initialization: **He (Kaiming) initialization** for the shallow layers and **Orthogonal initialization** for deeper layers. The training process uses data augmentation, batch normalization, cosine annealing learning rate scheduling, and Adam optimizer.

## Features

- Custom VGG-like CNN architecture
- Hybrid weight initialization (He + Orthogonal)
- Data augmentation (random cropping and flipping)
- Batch normalization
- Cosine annealing learning rate scheduler
- CIFAR-10 dataset support
- Training and validation accuracy/loss plots

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib

Install dependencies with:

```bash
pip install torch torchvision matplotlib
```

## Files

- `main.py`: Main training script (the code you provided)
- CIFAR-10 dataset is automatically downloaded via `torchvision.datasets`

## How to Run

1. Clone this repository or copy the code into `cifar10(orth+He)-VGG.py`.

2. Run the training script:

```bash
python cifar10(orth+He)-VGG.py
```

3. After training, the script will display loss and accuracy plots.

## Description of Key Components

- **VGGLike(nn.Module)**: Custom CNN model similar to VGG with two convolutional blocks.
- **custom_init(model)**: Applies He initialization to early layers and Orthogonal to deeper ones.
- **train_model(model)**: Handles training, validation, and plotting metrics over epochs.

---



**DIRECTORY INFO-**
```bash
Final/cifar10(orth+He)-VGG.py
```
---




**4.3 cifar10(orth+he).ipynb**


---

# CIFAR-10 Classification with Custom Initialization

This project implements a deep 9-layer Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. It combines **He initialization for shallow layers** and **Orthogonal initialization with ReLU gain for deeper layers** to optimize training performance. The model is trained with data augmentation, batch normalization, dropout, and a cosine annealing learning rate scheduler.

## Features

- Custom hybrid initialization (He + Orthogonal with gain)
- 9-layer CNN architecture with BatchNorm, ReLU, and Dropout
- Cosine annealing learning rate scheduler
- Real-time plotting of training/validation accuracy and loss
- GPU support with CUDA

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Prerequisites

Make sure you have the following installed:

```bash
pip install torch torchvision matplotlib
```

## How to Run

1. Clone the repository or copy the code to a Python file (e.g., `cifar10(orth+He)-VGG.py`).

2. Run the script:

```bash
python cifar10(orth+He)-VGG.py
```

3. The model will begin training for 50 epochs, and after completion, it will show loss and accuracy plots for training and validation data.

## Notes

- The training is performed on GPU if available.
- Model uses data augmentation (random crop and horizontal flip).
- Uses Cosine Annealing LR scheduling for smooth convergence.
- Default learning rate: `0.001`, Batch size: `128`, Epochs: `50`.

---

**DIRECTORY INFO-**
```bash
Final/cifar10(orth+he).ipynb
```
---









