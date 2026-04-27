import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

# =========================
# MODEL LINK (DISPLAY ONLY)
# =========================
MODEL_HF_LINK = "https://huggingface.co/Mudassir-08/lenet5-mnist"

# =========================
# MODEL ARCHITECTURE
# =========================
class LeNet5(torch.nn.Module):
    def __init__(self, num_classes=10, grayscale=True):
        super().__init__()
        in_channels = 1 if grayscale else 3

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# =========================
# LOAD MODEL FROM HF HUB
# =========================
repo_id = "Mudassir-08/lenet5-mnist"

model_path = hf_hub_download(
    repo_id=repo_id,
    filename="lenet5_model.pth"
)

checkpoint = torch.load(model_path, map_location="cpu")

model = LeNet5(num_classes=checkpoint["num_classes"], grayscale=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# TRANSFORM (MATCH TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# PREDICTION
# =========================
def predict(image):
    image = image.convert("L")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)

    return f"Predicted Digit: {pred.item()}"

# =========================
# UI FUNCTION
# =========================
def ui_predict(image):
    return predict(image)

# =========================
# GRADIO INTERFACE
# =========================
gr.Interface(
    fn=ui_predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title=" LeNet-5 MNIST Digit Classifier",
    description="""
 Model: LeNet-5 CNN trained on MNIST  
 Test Accuracy: ~98.82%  

 IMPORTANT IMAGE REQUIREMENTS:
- Must be a handwritten digit (0–9)
- Prefer black digit on white background
- Single digit only (not multiple numbers)
- Avoid colorful images or noisy backgrounds
- Image should be similar to MNIST style

 Model Download:
https://huggingface.co/Mudassir-08/lenet5-mnist
"""
).launch()
