import torch
import torchvision
from torch.utils.data import DataLoader
from model import NET
from torch.utils.tensorboard import SummaryWriter

# Get data
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

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"The length of train data : {train_data_size}")

print(f"The length of test data : {test_data_size}")

# loader data
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# Build model
model = NET()

print(model)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
# # record number of training
# total_train_step = 0
# # record number the test
# total_test_step = 0
# training number
epochs = 30
# add tensorboard

writer = SummaryWriter("./logs")

for epoch in range(epochs):
    running_loss = 0.0
    # model train
    model.train()
    for batch_idx, data in enumerate(train_loader, 0):
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # optimizer work
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"epoch : {epoch}, loss : {running_loss/len(train_loader)}")
    writer.add_scalar("train_loss", running_loss / len(train_loader), epoch)
    # model test
    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            total_accuracy += (predicted == targets).sum().item()

    print(
        f"test result: epoch : {epoch}, loss : {total_test_loss/len(test_loader)}, accurancy rate : {total_accuracy / len(test_loader)}"
    )

    writer.add_scalar("test_train", total_test_loss / len(test_loader), epoch)
    writer.add_scalar("test_accurancy", total_accuracy / len(test_loader), epoch)
    torch.save(model, "./model/model_{}.pth".format(epoch))
writer.close()
