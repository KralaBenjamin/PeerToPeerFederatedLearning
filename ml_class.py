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
                 learning_rate=0.001, num_epochs=2, max_n=5, batch_size=64,
                 num_classes=3, num_train_samples=128, num_test_samples=512):

        if train_dataloader is None or test_dataloader is None:
            random_classes = random.sample(range(10), num_classes)      # select random classes

            # Load full train and test datasets
            train_dataset_full = torchvision.datasets.FashionMNIST(root='./data',
                                                                   train=True,
                                                                   transform=transforms.Compose([
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.5,), (0.5,))
                                                                   ]),
                                                                   download=True)

            test_dataset_full = torchvision.datasets.FashionMNIST(root='./data',
                                                                  train=False,
                                                                  transform=transforms.Compose([
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize((0.5,), (0.5,))
                                                                  ]),
                                                                  download=True)

            train_dataset = []
            test_dataset = []
            included_train_indices = []
            included_test_indices = []
            train_index = 0
            test_index = 0

            # sample from train dataset without duplicates and only classes from above random step
            while train_index < num_train_samples:
                random_sample_index = random.choice(range(len(train_dataset_full)))

                # skip duplicates
                if random_sample_index in included_train_indices:
                    continue

                # skip other classes
                (_, data) = train_dataset_full.__getitem__(random_sample_index)
                if data not in random_classes:
                    continue

                # add to train dataset
                train_dataset.append(train_dataset_full.__getitem__(random_sample_index))
                included_train_indices.append(random_sample_index)
                train_index += 1

            print(f"Created train dataset with classes {random_classes} of size {len(train_dataset)}")

            # Do the same for test dataset
            # TODO should the test dataset contain all classes?
            while test_index < num_test_samples:
                random_sample_index = random.choice(range(len(test_dataset_full)))

                if random_sample_index in included_test_indices:
                    continue

                test_dataset.append(test_dataset_full.__getitem__(random_sample_index))
                included_test_indices.append(random_sample_index)
                test_index += 1

            print(f"Created test dataset with classes {random_classes} of size {len(train_dataset)}")

            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True)

            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=True)
            self.classes = self.get_classes(train_dataset)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = LeNet()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_n = max_n

        self.test_results = []

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
        # Record test results after each training step
        self.test()

    def test(self):
        # Testen des Modells
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_dataloader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total

            print(f"Genauigkeit des Modells auf Testdaten: {100*accuracy} ")
            self.test_results.append(accuracy)

        return accuracy

    def average(self, new_statedicts):
        # Append weights by current own weights
        own_statedict = copy.deepcopy(self.model.state_dict())

        # Iterate over all keys
        for key in own_statedict:
            # Iterate over all additional statedicts
            for statedict in new_statedicts:
                own_statedict[key] += statedict[key]
            own_statedict[key] = torch.divide(own_statedict[key], len(new_statedicts) + 1)

        # load averaged statedict in new model and train again
        new_model = LeNet()
        new_model.load_state_dict(own_statedict)
        self.model = new_model
        self.train()

    def select_max(self, new_statedicts):
        own_statedict = copy.deepcopy(self.model.state_dict())
        new_statedicts.append(own_statedict)

        # TODO select maximal scored state_dict


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
