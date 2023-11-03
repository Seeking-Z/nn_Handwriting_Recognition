import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

train_Data = datasets.MNIST(
    root='./mnist/',
    train=True,
    download=True,
    transform=transform
)

test_Data = datasets.MNIST(
    root='./mnist/',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_Data, shuffle=True, batch_size=64)
test_loader = DataLoader(test_Data, shuffle=False, batch_size=64)


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, the_x):
        return self.net(the_x)


model = DNN().to('cuda:0')
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.5
)

epochs = 30
losses = []

for epoch in range(epochs):
    for (x, y) in train_loader:
        x, y = x.to('cuda:0'), y.to('cuda:0')
        Pred = model(x)
        loss = loss_fn(Pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Fig = plt.figure()
plt.plot(range(len(losses)), losses)
plt.show()

correct = 0
total = 0
with torch.no_grad():
    for (x, y) in test_loader:
        x, y = x.to('cuda:0'), y.to('cuda:0')
        Pred = model(x)
        _, predicted = torch.max(Pred.data, dim=1)
        correct += torch.sum((predicted == y))
        total += y.size(0)

print(f'测试集准确率为：{100 * correct / total}')
torch.save(model.state_dict(), 'model.pth')
