# LeNet-5 MNIST Digit Classifier (PyTorch)

##  Overview
This project implements the classic **LeNet-5 Convolutional Neural Network** using PyTorch for handwritten digit classification on the MNIST dataset.  
It includes training, evaluation, and deployment via a Hugging Face Space.

---

##  Objective
Classify handwritten digits (0–9) using a lightweight CNN with high accuracy and efficient training.

---

##  Live Demo (Inference)
https://huggingface.co/spaces/Mudassir-08/lenet5-mnist-demo

Upload an image and get real-time digit prediction.

---

##  Trained Model (Hugging Face Hub)
https://huggingface.co/Mudassir-08/lenet5-mnist-demo

Download the trained `lenet5_model.pth` from here for inference or fine-tuning.

---

##  Model Architecture
- Conv2D → Tanh → MaxPool  
- Conv2D → Tanh → MaxPool  
- Fully Connected: 120 → 84 → 10  
- Activation: Tanh  
- Loss: CrossEntropyLoss  
- Optimizer: SGD (lr = 0.1)

---

##  Dataset
- MNIST handwritten digits dataset  
- 60,000 training images  
- 10,000 test images  
- Image size: 28×28 (resized to 32×32)  
- Grayscale images (1 channel)

---

##  Results
| Metric | Value |
|--------|--------|
| Train Accuracy | 99.30% |
| Validation Accuracy | 98.98% |
| Test Accuracy | 98.82% |

---

##  Project Structure
LeNet-5/

├── src/

│   ├── model.py

│   ├── helper_train.py

│   ├── helper_dataset.py

│   ├── helper_plotting.py

│   └── helper_evaluation.py


├── saved_models/

│   └── lenet5_model.pth


├── app.py

├── requirements.txt

└── README.md

---

##  Requirements
torch
torchvision
gradio
pillow
numpy
matplotlib

---

##  Features
- Classic LeNet-5 CNN implementation
- High accuracy (~98.8% on MNIST test set)
- Fast training convergence
- Modular PyTorch codebase
- Deployment-ready Gradio app
- End-to-end ML pipeline (training → evaluation → inference)

---

##  Pipeline
Image → Preprocessing → CNN Model → Digit Prediction

---

##  Author
Malik Mudassir Iqbal  
Deep Learning Researcher 
