from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import torch
from models.model import load_model, load_class_names
from utils.preprocessing import preprocess_image
from utils.inference import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Folder to store uploaded images

# Define device for PyTorch (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ImageNet class names
class_names = load_class_names()

# Initialize the model
model = load_model('models/model.pth', device=device)
print("Model loaded successfully.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return 'No file part', 400
        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image and make a prediction (get top prediction)
            input_tensor = preprocess_image(filepath)
            top_prediction = predict(model, input_tensor, device, class_names)

            # Render the result page with the top prediction
            return render_template('result.html', top_prediction=top_prediction, image_file=filename)

    return render_template('index.html')


@app.route('/feedback', methods=['POST'])
def feedback():
    # Handle feedback logic here
    feedback = request.form['feedback']
    image_path = request.form['image_path']
    predicted_label = request.form['predicted_label']

    if feedback == 'correct':
        return render_template('thanks.html')
    else:
        # User says the prediction was wrong, ask for the correct label
        return render_template('correct_label_form.html', image=image_path, predicted_label=predicted_label)


@app.route('/update_model', methods=['POST'])
def update_model():
    correct_label = request.form['correct_label']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], request.form['image_path'])

    # Preprocess the image and convert the correct label to tensor
    input_tensor = preprocess_image(image_path)
    label_idx = class_names.index(correct_label)
    label_tensor = torch.tensor([label_idx]).to(device)

    # Fine-tune the model (unfreeze final layer)
    model.train()
    for param in model.model.fc.parameters():
        param.requires_grad = True

    # Define optimizer and loss function for fine-tuning
    optimizer = torch.optim.SGD(model.model.fc.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Fine-tune the model
    optimizer.zero_grad()
    outputs = model(input_tensor.to(device))
    loss = criterion(outputs, label_tensor)
    loss.backward()
    optimizer.step()

    print(f"Model fine-tuned, loss: {loss.item()}")

    # Save the updated model
    torch.save(model.state_dict(), 'models/model.pth')
    print("Model saved successfully to 'models/model.pth'")

    return render_template('thanks.html')


if __name__ == '__main__':
    app.run(debug=True)
