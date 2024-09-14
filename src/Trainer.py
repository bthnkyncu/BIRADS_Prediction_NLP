import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train_model(self):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        for texts, labels in self.train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
        return running_loss / len(self.train_loader), correct_predictions / len(self.train_loader.dataset)

    def evaluate_model(self):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        with torch.no_grad():
            for texts, labels in self.test_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
        return running_loss / len(self.test_loader), correct_predictions / len(self.test_loader.dataset)