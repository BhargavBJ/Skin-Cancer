# Skin Cancer Classification using HAM10000 Dataset🧬

This project leverages deep learning (CNNs) to classify skin lesions using the **HAM10000** dataset. It aims to assist in the early detection of skin cancer by identifying various types of skin lesions from dermatoscopic images.

---

## 🔍 Project Overview

Skin cancer is among the most common types of cancer. Accurate early detection using deep learning can improve diagnosis and treatment planning. In this project:

- We use the **HAM10000** dataset with over 10,000 dermatoscopic images.
- The images are classified into 7 categories of skin lesions.
- A Convolutional Neural Network (CNN) is trained to perform multi-class classification.

---

## 📁 Dataset: HAM10000

The **"Human Against Machine with 10000 training images"** (HAM10000) dataset contains high-resolution images of pigmented skin lesions.

- Source: [Kaggle – HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Categories:
  - Melanocytic nevi (nv)
  - Melanoma (mel)
  - Benign keratosis-like lesions (bkl)
  - Basal cell carcinoma (bcc)
  - Actinic keratoses (akiec)
  - Vascular lesions (vasc)
  - Dermatofibroma (df)

---

## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab or Jupyter Notebook

---

## 🧠 Model Details

- **Architecture**: Pre-trained **ResNet50** model from ImageNet.
- **Transfer Learning**:
  - Base layers frozen initially
  - Custom top layers added for classification
- **Training Strategy**:
  - Data augmentation
  - Categorical cross-entropy loss
  - Adam optimizer
  - Learning rate tuning
---
### 📜Model evaluation:
- Accuracy
- Confusion matrix
- Precision, Recall, F1-score
- Visual predictions

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/BhargavBJ/Skin-Cancer-MNIST-HAM10000.git
   cd Skin-Cancer-MNIST-HAM10000
2. **Install dependencies**
(Optional if using Google Colab)

   ```bash
   pip install -r requirements.txt

Download the Dataset

From Kaggle HAM10000

Place the files in the project directory under /input/

Run the notebook
Open Skin_Cancer_MNIST_HAM10000.ipynb using Jupyter Notebook or Colab and run all cells.
