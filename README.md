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





