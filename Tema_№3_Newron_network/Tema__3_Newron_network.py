import torch
import matplotlib.pyplot as plt

def predict(net, x, y):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', label='Ground truth')
    plt.plot(x.numpy(), y_pred.detach().numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

def target_function(x):
    return 2**x * torch.sin(2**-x)

def metric(pred, target):
    return (pred - target).abs().mean()

class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

net = RegressionNet(50)  

x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  

for epoch_index in range(4450):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = metric(y_pred, y_train)
    loss_value.backward()
    optimizer.step()

    with torch.no_grad():
        validation_error = metric(net.forward(x_validation), y_validation)
    
    if (epoch_index + 1) % 50 == 0:
        print(f'Epoch {epoch_index + 1}, Train Loss: {loss_value.item():.4f}, Validation Error: {validation_error.item():.4f}')


predict(net, x_validation, y_validation)