import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

class LinearEvaluator:
    def __init__(self,encoder,feature_size,device):
        self.encoder =encoder
        self.device = device

        self.encoder.to(self.device)

        #freeze the encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        #simple classifier
        self.classifier = nn.Linear(
            feature_size,
            10
        )
        self.classifier.to(self.device)

    def train(self,train_loader,epochs=5,learning_rate =0.001):
        optimizer = Adam(
            self.classifier.parameters(),
            lr = learning_rate
        )
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(1,epochs + 1):
            self.classifier.train()
            self.encoder.eval()

            for images,labels in tqdm(
                train_loader,
                desc = f"Evaluation {epoch}/{epochs}"
            ):

                images = images.to(self.device)
                labels = labels.to(self.device)

                #encoder is frozen,so we do not calculate gradients
                with torch.no_grad():
                    features = self.encoder(images)

                predictions = self.classifier(
                    features
                )
                loss = loss_function(
                    predictions,
                    labels
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

    def test(self,test_loader):
        self.encoder.eval()
        self.classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images,labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.encoder(images)

                predictions = self.classifier(
                    features
                )
                predicted_classes = predictions.argmax(
                    dim = 1
                )

                correct +=(
                    predicted_classes == labels
                ).sum().item()

                total += labels.size(0)
                accuracy = correct / total

                return accuracy
