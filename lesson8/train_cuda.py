import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import NET, MNIST_CNN, LinearRegression, LinearRegression2
from torch.utils.tensorboard import SummaryWriter
import time

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : { device}")
# 1 Prepare data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

# print(type(train_dataset))

# # access a single image label
# image, label = train_dataset[0]
# print(f"label : {label}")
# # The picture is 1*28*28, we can convert it to 28*28 matrix to diplay
# image = image.squeeze(0)  # Remove the first dimension (1) for channels, leaving 28x28
# plt.imshow(image.reshape(28, 28))
# plt.title(f"label : {label}")
# plt.show()

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(len(train_dataset))
print(len(train_loader))


# Instantiate Model Class
# input_dim = 28 * 28  # size of image px*px
# output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9
# model = LinearRegression2(input_dim, output_dim)

# loss_fn = torch.nn.CrossEntropyLoss()
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# epoch_num = 10

# loss_list = []
# for epoch in range(epoch_num):
#     loss_sum = 0.0
#     for batch_idx, data in enumerate(train_loader):
#         inputs, targets = data
#         outputs = model(inputs)
#         loss = loss_fn(outputs, targets)
#         loss_sum += loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"epoch : {epoch} loss : {loss_sum/len(train_loader)}")
#     loss_list.append(loss_sum / len(train_loader))

# plt.plot(range(epoch_num), loss_list)
# plt.show()


model = NET().to(device)

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

writer = SummaryWriter("./logs")
# start time for traing
t_start = time.time()
epoch_num = 10
loss_list = []
accuracy_num = 0.0
for epoch in range(epoch_num):
    loss_sum = 0
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()  # Accumulate the loss for the epoch
    t_stop = time.time()
    print(
        f"epoch : {epoch}; Time elapsed :{t_stop - t_start}; loss : {loss_sum/len(train_loader)}"
    )
    loss_list.append(loss_sum / len(train_loader))
    writer.add_scalar("train_loss", loss_sum / len(train_loader), epoch)

    # Evaluation mode for testing
    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            total_accuracy += (predicted == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_accuracy / len(test_dataset)

    writer.add_scalar("test_loss", avg_test_loss, epoch)
    writer.add_scalar("test_accuracy", accuracy, epoch)
    print(
        f"test eoch : {epoch} , Loss : {avg_test_loss :.4f}, Accuarcy : {accuracy * 100 :.2f} "
    )
    torch.save(model.state_dict(), f"./model/model_epoch_{epoch}.pth")
# plt.plot(range(epoch_num), loss_list)
# plt.show()
# Close TensorBoard writer
writer.close()
