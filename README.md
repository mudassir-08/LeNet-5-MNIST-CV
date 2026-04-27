# LeNet-5 MNIST Digit Classification using PyTorch  
### End-to-End Computer Vision System (Training → Evaluation → Deployment)

---

##  Project Deployment & Access

I have developed, trained, and deployed this complete **LeNet-5 based digit classification system** using PyTorch and Hugging Face Spaces.

### 🔹 Live Inference System (Deployed Model)
You can directly test the trained model here:

-- https://huggingface.co/spaces/Mudassir-08/lenet5-mnist-demo  

✔ This interface allows you to:
- Upload handwritten digit images  
- Run real-time inference using the trained CNN  
- View predicted digit outputs instantly  

---

### 🔹 Trained Model (Reusable Artifact)
The trained model is publicly hosted and can be reused for research or applications:

-- https://huggingface.co/Mudassir-08/lenet5-mnist-demo  

✔ You can:
- Download the `lenet5_model.pth` checkpoint  
- Load it in PyTorch for inference  
- Fine-tune it on new datasets  
- Integrate it into APIs or production systems  

---

## 1. Project Overview

This project implements a full **computer vision pipeline** using the classical **LeNet-5 Convolutional Neural Network (CNN)** architecture.

The goal was to build not just a model, but a **complete deep learning system**, including:

- Data preprocessing (MNIST dataset handling)
- CNN model implementation (LeNet-5)
- Training & optimization pipeline
- Evaluation with quantitative + qualitative analysis
- Deployment as a working inference application

This project demonstrates how foundational CNN architectures can be engineered into real-world deployable systems.

---

## 2. What I Built

This is a complete end-to-end deep learning system that includes:

### 🔹 Model Development
- Implemented LeNet-5 architecture in PyTorch  
- Designed for grayscale digit classification (10 classes)  
- Adapted input pipeline to 32×32 resolution  

### 🔹 Training Pipeline
- SGD optimizer with learning rate scheduling  
- Batch training with validation monitoring  
- Reproducible experiments using fixed random seeds  

### 🔹 Evaluation System
- Accuracy tracking across train/validation/test sets  
- Confusion matrix-based error analysis  
- Visual inspection of model predictions  

### 🔹 Deployment System
- Converted trained model into deployable artifact  
- Built Hugging Face Space for real-time inference  
- Enabled image upload → prediction pipeline  

---

## 3. Model Architecture (LeNet-5)
Input Image (1 × 32 × 32)

→ Conv Layer (6 filters) + Tanh + Pooling

→ Conv Layer (16 filters) + Tanh + Pooling

→ Fully Connected (120)

→ Fully Connected (84)

→ Output Layer (10 classes)


This architecture is a foundational CNN model that introduced hierarchical feature learning in computer vision.

---

## 4. Dataset Used

- MNIST handwritten digit dataset  
- 60,000 training samples  
- 10,000 test samples  
- 10 output classes (digits 0–9)  
- Grayscale images resized to 32×32  

---

## 5. Performance Summary

The trained model achieves strong results on the MNIST benchmark:

| Metric | Score |
|--------|------|
| Training Accuracy | 99.30% |
| Validation Accuracy | 98.98% |
| Test Accuracy | 98.82% |

### Key Observations:
- Fast convergence during early epochs  
- Stable generalization on unseen data  
- Very low misclassification rate  
- Strong performance for a lightweight CNN  

---

## 6. System Pipeline

The full system follows this workflow:
Image Input → Preprocessing → LeNet-5 CNN → Feature Extraction → Classification → Digit Output.


This pipeline is deployed and accessible via the Hugging Face interface.

---

## 7. Technical Highlights

- Full PyTorch implementation from scratch  
- Modular training and evaluation pipeline  
- Real-time inference deployment using Hugging Face Spaces  
- Confusion matrix and visualization-based evaluation  
- Reproducible experimental setup  
- Production-ready model packaging  

---

## 8. Conclusion

This project demonstrates the complete lifecycle of a deep learning system — from model design and training to real-world deployment.

By implementing LeNet-5 and deploying it as an interactive application, this work bridges the gap between theoretical computer vision models and practical machine learning systems.

---

## 9. Author

**Malik Muhammad Mudassir Iqbal**  
Deep Learning & Computer Vision Engineer  