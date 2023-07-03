import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset



class MLModell():
    # handles all ML stuff


    def __init__(self, train_dataloader, test_dataloader, 
                 learning_rate = 0.001, num_epochs = 1, max_n = 5, ):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = LeNet()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_n = max_n


    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_dataloader):

                if self.max_n and i >= self.max_n:
                    continue

                # Vorwärtsdurchlauf
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Rückwärtsdurchlauf und Optimierung
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch [{}/{}], Schritt [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                    




# LeNet Modell Definition
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(2450, 120) #800
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, stride=2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def get_filtered_dataloader(dataset, classes):
    filtered_list = [
        i for i, (x, y) in enumerate(dataset) if y in classes
    ]
    return torch.utils.data.Subset(dataset, filtered_list)
    
        