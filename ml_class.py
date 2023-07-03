import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import copy


class MLModell:
    # handles all ML stuff

    def __init__(self, train_dataloader=None, test_dataloader=None,
                 learning_rate=0.001, num_epochs=1, max_n=5, ):

        # TODO replace by selection based on classes: random data of only n classes available
        if train_dataloader is None or test_dataloader is None:
            train_dataset_full = torchvision.datasets.FashionMNIST(root='./data',
                                                              train=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor()
                                                              ]),
                                                              download=True)

            train_dataset, _ = torch.utils.data.random_split(train_dataset_full, [20, len(train_dataset_full)-20])

            test_dataset_full = torchvision.datasets.FashionMNIST(root='./data',
                                                             train=False,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor()
                                                             ]),
                                                             download=True)

            test_dataset, _ = torch.utils.data.random_split(test_dataset_full, [20, len(test_dataset_full)-20])

            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=64,
                                                           shuffle=True)

            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=64,
                                                          shuffle=True
                                                          )
            self.classes = self.get_classes(train_dataset)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = LeNet()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_n = max_n

    def get_classes(self, train_dataset):
        classes_set = set()
        for _, data in train_dataset:
            classes_set.add(data)
        return list(classes_set)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_dataloader):

                if self.max_n and i >= self.max_n:
                    continue

                # Vorwärtsdurchlauf
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Rückwärtsdurchlauf und Optimierung
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch [{}/{}], Schritt [{}/{}], Loss: {:.4f}'.format(epoch + 1,
                                                                            self.num_epochs, i + 1,
                                                                            len(list(self.train_dataloader)),
                                                                            loss.item()))

    def average(self, new_statedicts):
        # Append weights by current own weights
        own_statedict = copy.deepcopy(self.model.state_dict())

        # Iterate over all keys
        for key in own_statedict:
            # Iterate over all additional statedicts
            for statedict in new_statedicts:
                own_statedict[key] += statedict[key]
            own_statedict[key] = torch.divide(own_statedict[key], len(new_statedicts) + 1)

        # load new statedict and train again
        self.model.load_state_dict(own_statedict)
        self.train()

    def get_current_weights(self):
        if self.model is not None:
            return self.model.state_dict()


# LeNet Modell Definition
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(2450, 120)  # 800
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
