import torch
import torch.nn as nn
from torchvision import models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()

        # Load resNet18
        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)


def load_model(model_path, num_classes, device):
    model = ImageClassifier(num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    model.to(device)
    
    return model

def predict(model, input_tensor, device, class_names):
    model.eval()
    
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        
        _, predicted = torch.max(output, 1)
    
    # Return the class name
    return class_names[predicted.item()]