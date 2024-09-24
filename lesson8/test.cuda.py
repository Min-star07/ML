import torch
import torchvision
from PIL import Image
import cv2
from model import NET, MNIST_CNN, LinearRegression, LinearRegression2

class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print(f"Using device {device}")

img = cv2.imread("./test8.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
img = Image.fromarray(img)
img_trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
    ]
)
img = img_trans(img)
print(img.shape)
img = torch.reshape(img, (1, 1, 28, 28))
img = img.to(device)

model = NET().to(device)

model.load_state_dict(torch.load("./model/model_epoch_9.pth"))
model.eval()
with torch.no_grad():
    outputs = model(img)
    print(f"Model outputs : {outputs}")

    _, predicted = torch.max(outputs.data, dim=1)

predicted_class_idx = predicted.item()
predicted_class_item = class_names[predicted_class_idx]


print(f"Predicted class index: {predicted_class_idx}")
print(f"Predicted class name: {predicted_class_item}")
