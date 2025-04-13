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
