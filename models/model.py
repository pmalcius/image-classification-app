import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import json
import os

# Load ImageNet class names from a JSON file
def load_class_names(file_path='imagenet-simple-labels.json'):
    with open(file_path, 'r') as f:
        class_names = json.load(f)
    return class_names

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        # Load the pre-trained ResNet-18 model
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, x):
        return self.model(x)

def load_model(model_path=None, device='cpu'):
    model = ImageClassifier()
    model.to(device)
    # If a saved model exists, load its state
    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print("Using pre-trained ResNet-18 model.")
    return model
