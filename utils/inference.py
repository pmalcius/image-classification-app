import torch

def predict(model, input_tensor, device, class_names):
    """
    Perform a prediction using the provided model and input tensor.
    Returns the top predicted class name and its probability.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move the input tensor to the same device as the model (CPU or GPU)
    input_tensor = input_tensor.to(device)
    
    # Perform the forward pass with no gradient calculation (for inference)
    with torch.no_grad():
        output = model(input_tensor)
        
        # Get the top predicted class index and its probability
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_idx = torch.topk(probabilities, 1, dim=1)
    
    # Return the top class name and its probability
    top_prediction = (class_names[top_idx.item()], top_prob.item())
    
    return top_prediction
