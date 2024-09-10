import torch


# Build NET
class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=1, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, stride=1, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, stride=1, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 64),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# if __name__ == "__main__":
#     model = NET()
#     input = torch.ones((64, 3, 32, 32))
#     output = model(input)
#     print(output.shape)
 