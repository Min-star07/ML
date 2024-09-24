import torch
import torchvision
from PIL import Image
import cv2
from model import NET  # Assuming you have this model implemented in a separate file

# CIFAR-10 class names for reference
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Check if CUDA (GPU) is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the image using OpenCV
img = cv2.imread("./img/dog.png")  # Ensure the image path is correct
img = cv2.cvtColor(
    img, cv2.COLOR_BGR2RGB
)  # Convert BGR to RGB (OpenCV uses BGR by default)

# Convert the OpenCV image (NumPy array) to a PIL image
img = Image.fromarray(img)

# Define transformations: Resize to (32, 32) and convert to tensor
img_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),  # Resize the image to 32x32
        torchvision.transforms.ToTensor(),  # Convert the image to a tensor
    ]
)

# Apply the transformations to the image
img = img_transforms(img)

# Reshape the image to include the batch dimension (1, 3, 32, 32)
img = torch.reshape(img, (1, 3, 32, 32))

# Move the image tensor to the device (GPU or CPU)
img = img.to(device)

# Check the transformed image's shape
print(
    f"Image shape after transformation: {img.shape}"
)  # Should output: torch.Size([1, 3, 32, 32])

# Load the pre-trained model and move it to the device
model = NET().to(device)  # Initialize the model

# Load the model's state dictionary (pre-trained weights)
model.load_state_dict(torch.load("./model/model_epoch_99.pth"))

# Set the model to evaluation mode (for inference)
model.eval()

# Perform inference (without computing gradients)
with torch.no_grad():
    outputs = model(img)  # Forward pass through the model
    print(f"Model outputs (logits): {outputs}")

    # Get the predicted class (the class with the highest score)
    _, predicted = torch.max(outputs.data, 1)

# Output the predicted class index and class name
predicted_class_idx = predicted.item()  # Extract the class index
predicted_class_name = class_names[
    predicted_class_idx
]  # Map the index to the class name

print(f"Predicted class index: {predicted_class_idx}")
print(f"Predicted class name: {predicted_class_name}")
