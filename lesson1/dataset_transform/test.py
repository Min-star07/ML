import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=True, transform=dataset_transform, download=False
)
test_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, transform=dataset_transform, download=False
)
print(test_set[0])
print(test_set.classes)

# writer = SummaryWriter("logs")
# img, target = test_set[0]
# writer.add_image("test_set", img, 0)
# # print(img)
# # print(target)
# # img.show()
# writer.close()


from torch.utils.data import DataLoader

test_loader = DataLoader(
    test_set, batch_size=64, shuffle=True, num_workers=2, drop_last=False
)
print(test_loader)

# img, target = test_set[0]
# print(img.shape)
# print(target)
writer = SummaryWriter("datalogs")
for i, data in enumerate(test_loader):
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_set", imgs, i, dataformats="NCHW")

writer.close()

writer = SummaryWriter("datalogs")
for epoch in range(2):
    for i, data in enumerate(test_loader):
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("epoch:{}".format(epoch), imgs, i, dataformats="NCHW")

    writer.close()
