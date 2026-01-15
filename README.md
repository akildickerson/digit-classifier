# MNIST Digit Classifier (PyTorch)

A simple neural network built with **PyTorch** to classify handwritten digits from the **MNIST** dataset (0–9).

✅ **Test Accuracy:** **89%**

---

## 📌 Overview

This project uses a fully-connected feedforward neural network (MLP) to recognize handwritten digits.  
Each MNIST image is **28×28 grayscale**, which is flattened into a **784-length vector** and passed through dense layers to predict the digit class.

---

## 🧠 Model

### Architecture
- **Input:** 784 features (28×28 image flattened)
- **Hidden Layers:** 2 dense layers with 512 neurons each
- **Activation:** ReLU
- **Output:** 10 logits (digits 0–9)
