import torch as th
import torch.nn as nn
import torchmetrics.functional as metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from src.models import ResNet


if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder("data/BinaryClassification", transform=transform)
    train_size = int(0.8 * len(dataset))
    trainset, testset = random_split(dataset, [train_size, len(dataset) - train_size])

    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

    model = ResNet(1).to(device)

    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = []
    
    for epoch in tqdm(range(50), desc="Training model", leave=False):
        iteration_losses = []
        iteration_auc = []

        for inputs, labels in tqdm(trainloader, desc="Iterating over training data", leave=False):
            
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            logits = sigmoid(outputs).view(-1)

            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            iteration_losses.append(loss.item())
            iteration_auc.append(metrics.auc(logits.detach().cpu(), labels.detach().cpu(), reorder=True))
        
        epoch_losses.append(sum(iteration_losses) / len(iteration_losses))


        plt.figure(figsize=(10, 8))
        plt.plot(range(len(trainloader)), iteration_losses, label="Loss")
        plt.plot(range(len(trainloader)), iteration_auc, label="AUC")
        plt.xlabel("Iteration (epoch: {})".format(epoch))
        plt.ylabel("Loss")
        plt.savefig(f"figures/binary-classification/training-metrics-{epoch}.png")

    model.eval()

    with th.no_grad():
        test_losses = []
        test_auc = []
        test_recall = []

        for inputs, labels in tqdm(testloader, desc="Iterating over test data", leave=False):
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)
            logits = sigmoid(outputs).view(-1)

            loss = criterion(logits, labels)

            test_losses.append(loss.item())
            test_auc.append(metrics.auc(logits, labels, reorder=True))
            test_recall.append(metrics.recall(logits, labels))

        print(f"Test loss: {sum(test_losses) / len(test_losses)}")
        print(f"Test AUC: {sum(test_auc) / len(test_auc)}")
        print(f"Test recall: {sum(test_recall) / len(test_recall)}")

        plt.figure(figsize=(10, 8))
        plt.plot(range(len(testloader), test_losses), label="Loss")
        plt.plot(range(len(testloader), test_auc), label="AUC")
        plt.plot(range(len(testloader), test_recall), label="Recall")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("figures/binary-classification/test-metrics.png")
