import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


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


model = DNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()
model.to('cuda:0')

for i in range(1, 4):
    for i in range(1, 11):
        image = Image.open(f'./test/2-{i}.jpg')

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])

        image = ImageOps.invert(image)
        image = Image.eval(image, lambda z: 0 if z < 128 else z)

        input_image = transform(image).unsqueeze(0)
        # print(input_image[0])
        plt.imshow(image, cmap='gray')
        plt.show()

        with torch.no_grad():
            input_image = input_image.to('cuda:0')
            Pred = model(input_image)
            _, predicted = torch.max(Pred.data, dim=1)
            print(predicted.item(), end=' ')
    print()
    