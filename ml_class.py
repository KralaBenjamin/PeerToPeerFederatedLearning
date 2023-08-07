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
                 val_dataloader=None, val_global_dataloader=None,
                 learning_rate=0.001, num_epochs=2, max_n=5, batch_size=64,
                 num_classes=3, num_train_samples=128, num_test_samples=512, num_val_samples=1024):

        if train_dataloader is None or test_dataloader is None:
            random_classes = random.sample(range(10), num_classes)  # select random classes

            # Load full train and test datasets
            train_dataset_full = torchvision.datasets.FashionMNIST(root='./data',
                                                                   train=True,
                                                                   transform=transforms.Compose([
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.5,), (0.5,))
                                                                   ]),
                                                                   download=True)

            val_global_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                                   train=False,
                                                                   transform=transforms.Compose([
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.5,), (0.5,))
                                                                   ]),
                                                                   download=True)

            # All datasets are created from train data, so the test data can be used for the overall validation
            # The already used indices are passed to avoid any sample showing up twice in a local dataset
            train_dataset, train_indices = self.create_dataset(train_dataset_full, num_train_samples,
                                                               [], random_classes)
            test_dataset, test_indices = self.create_dataset(train_dataset_full, num_test_samples,
                                                             train_indices, random_classes)
            val_dataset, val_indices = self.create_dataset(train_dataset_full, num_val_samples,
                                                           test_indices)

            print(f"Created datasets with classes {random_classes}")

            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True)

            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=True)

            val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True)

            val_global_dataloader = torch.utils.data.DataLoader(dataset=val_global_dataset,
                                                                batch_size=batch_size,
                                                                shuffle=True)

            # set owned classes
            self.classes = self.get_classes(train_dataset)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.val_global_dataloader = val_global_dataloader

        self.model = LeNet()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_n = max_n

        self.val_results = []

    def create_dataset(self, full_dataset, num_samples, covered_indices, classes=None):
        sample_index = 0
        result_dataset = []
        while sample_index < num_samples:
            # select random index from dataset
            random_sample_index = random.choice(range(len(full_dataset)))

            # skip duplicates
            if random_sample_index in covered_indices:
                continue

            # skip other classes
            if classes is not None:
                (_, data) = full_dataset.__getitem__(random_sample_index)
                if data not in classes:
                    continue

            # add to train dataset
            result_dataset.append(full_dataset.__getitem__(random_sample_index))
            covered_indices.append(random_sample_index)
            sample_index += 1

        return result_dataset, covered_indices

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
        self.validate()

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

            print(f"Genauigkeit des Modells auf Testdaten: {100 * accuracy} ")

        return accuracy

    def test_model(self, state_dict):
        new_model = LeNet()
        new_model.load_state_dict(state_dict)
        new_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_dataloader:
                outputs = new_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total

            print(f"Genauigkeit des Modells auf Testdaten: {100 * accuracy} ")

        return accuracy

    def validate(self):
        # Testen des Modells
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.val_dataloader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total

            print(f"Genauigkeit des Modells auf lokaler Validation: {100 * accuracy} ")
            self.val_results.append(accuracy)

        return accuracy

    def validate_global(self):
        # Testen des Modells
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.val_global_dataloader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total

            print(f"Genauigkeit des Modells auf globaler Validation: {100 * accuracy} ")

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

    def select_max(self, state_dicts):
        own_state_dict = copy.deepcopy(self.model.state_dict())

        if len(state_dicts) == 0:
            state_dicts.append(own_state_dict)

        comp_list = [None] * len(state_dicts)

        for dict_index, state_dict in enumerate(state_dicts):
            comp_list[dict_index] = self.test_model(state_dict)

        max_dict_index = comp_list.index(max(comp_list))

        # load maximal statedict in new model and train again
        new_model = LeNet()
        new_model.load_state_dict(state_dicts[max_dict_index])
        self.model = new_model
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
