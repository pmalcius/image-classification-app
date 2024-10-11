from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path):
    # Define the transformations: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size ResNet expects
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Mean values for ImageNet-trained models
            std=[0.229, 0.224, 0.225]    # Standard deviation for ImageNet-trained models
        )
    ])
    
    # Open image, apply transformations
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor
