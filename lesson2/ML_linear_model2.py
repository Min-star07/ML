import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

car_prices_array = [3, 4, 5, 6, 7, 8, 9]
car_price_np = np.array(car_prices_array, dtype=np.float32)
car_price_np = car_price_np.reshape(-1, 1)
car_price_tensor = Variable(torch.from_numpy(car_price_np))
print(car_price_np)

# lets define number of car sell
number_of_car_sell_array = [7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array, dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1, 1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))
# print(number_of_car_sell_np)


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, ouput_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, ouput_size)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


input_dim = 1
output_dim = 1
model = LinearModel(input_dim, output_dim)
criterion = torch.nn.MSELoss()
# optimzer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

iteration_number = 1001
loss_list = []
for epoch in range(iteration_number):
    optimizer.zero_grad()
    y_pred = model(car_price_tensor)
    loss = criterion(y_pred, number_of_car_sell_tensor)
    # print(epoch, loss.item())
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()

print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())
# Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data.item())


plt.plot(range(iteration_number), loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()

predicted = model(car_price_tensor).data.numpy()
plt.scatter(
    car_prices_array, number_of_car_sell_array, label="original data", color="red"
)
plt.scatter(car_prices_array, predicted, label="predicted data", color="blue")

# predict if car price is 10$, what will be the number of car sell
# predicted_10 = model(torch.from_numpy(np.array([10]))).data.numpy()
# plt.scatter(10,predicted_10.data,label = "car price 10$",color ="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()
