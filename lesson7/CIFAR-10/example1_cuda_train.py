import torch
import torchvision
from torch.utils.data import DataLoader
from model import NET
from torch.utils.tensorboard import SummaryWriter
import time

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-10 training and test datasets
train_data = torchvision.datasets.CIFAR10(
    "../lesson3/dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.CIFAR10(
    "../lesson3/dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# Get sizes of training and testing datasets
train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"The length of train data: {train_data_size}")
print(f"The length of test data: {test_data_size}")

# Create data loaders for training and testing
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Build model and move it to GPU if available
model = NET().to(device)
print(model)

# Define the loss function (CrossEntropy for multi-class classification)
loss_fn = torch.nn.CrossEntropyLoss()

# Define the optimizer (SGD)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training parameters
epochs = 100

# TensorBoard writer
writer = SummaryWriter("./logs")

# Start timer for training
t_start = time.time()

# Training loop
for epoch in range(epochs):
    running_loss = 0.0  # Track training loss
    model.train()  # Set model to training mode

    # Train the model with training data
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        # Move inputs (imgs) and labels (targets) to GPU if available
        imgs, targets = imgs.to(device), targets.to(device)

        # Forward pass: compute model output
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)  # Compute the loss

        # Backpropagation and optimization
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss for the epoch

    # Measure time taken for epoch
    t_stop = time.time()
    print(
        f"Train epoch: {epoch}, Time elapsed: {t_stop - t_start:.2f}s, Loss: {running_loss / len(train_loader):.4f}"
    )

    # Log training loss to TensorBoard
    writer.add_scalar("train_loss", running_loss / len(train_loader), epoch)

    # Evaluation mode for testing
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():  # Disable gradient calculation for testing
        for imgs, targets in test_loader:
            # Move inputs (imgs) and labels (targets) to GPU if available
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)  # Compute test loss

            total_test_loss += loss.item()  # Accumulate test loss

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, dim=1)
            total_accuracy += (predicted == targets).sum().item()

    # Compute average test loss and accuracy
    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_accuracy / test_data_size

    # Log test loss and accuracy to TensorBoard
    writer.add_scalar("test_loss", avg_test_loss, epoch)
    writer.add_scalar("test_accuracy", accuracy, epoch)

    # Print test results
    print(
        f"Test epoch: {epoch}, Loss: {avg_test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%"
    )

    # Save the model checkpoint for the current epoch
    torch.save(model.state_dict(), f"./model/model_epoch_{epoch}.pth")

# Close TensorBoard writer
writer.close()
