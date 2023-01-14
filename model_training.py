import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
torch.cuda.empty_cache()
BS = 4
EPOCHS = 10
LR = 0.001
PATH = "CIFAR10-cnnModel.pth"
epoch_data = []
trainingLoss_data = []
validationLoss_data = []
validationAccuracy_data = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Pre defined values:")
print("Batch Size =", BS)
print("Epochs =", EPOCHS)
print("Learning Rate =", LR)
print("Path of the model:", PATH)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(256, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


cnn = CNN()
cnn.to(device)


def modelPlot():
    """
    Plots the model statistics sketching a graph for Validation Loss, Training Loss,
    and Validation Accuracy (y-axis) against Epochs (x-axis)
    """
    print("[INFO] Plotting the Model Statistics...")
    plt.plot(epoch_data, validationLoss_data, label = "Validation Loss", color = "r")
    plt.plot(epoch_data, trainingLoss_data, label = "Training Loss", color = "b")
    plt.ylabel("LOSS")
    plt.xlabel("EPOCHS")
    plt.legend()

    plt.figure()
    plt.plot(epoch_data, validationAccuracy_data, label = "Accuracy", color = "g")
    plt.ylabel("ACCURACY")
    plt.xlabel("EPOCHS")
    plt.legend()

    plt.show()
    print("...Done")


def modelAccuracy(testDataloader):
    """
    To calculate overall model accuracy at the end of training
    :param testDataloader: the Dataloader iterable of the testDataset
    """
    print("[INFO] Calculating Overall Model Accuracy...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testDataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = cnn(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("...Done")
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def loadCNN():
    """
    Loads the saved file from the specified PATH
    """
    print("[INFO] Loading Model...")
    cnn.load_state_dict(torch.load(PATH))
    print("...Done")


def saveModel():
    """
    Saves the file at the specified PATH
    """
    print("[INFO] Saving Model...")
    torch.save(cnn.state_dict(), PATH)
    print("...Done")


def accuracyCalculations(testDataloader, lossFunction):
    """
    Calculates the Accuracy After each Epoch

    :param lossFunction: the Cross Entropy Loss function
    :param testDataloader: the Dataloader iterable of the testDataset
    :return:
    """
    valAccuracy = 0.0
    valLoss = 0.0
    for i, data in enumerate(testDataloader):
        torch.cuda.empty_cache()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        loss = lossFunction(outputs, labels)
        loss.backward()
        valLoss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == labels).sum().item() / len(predicted)
        acc = acc * 100
        valAccuracy += acc
    i = float(i)
    valLoss_AVG = valLoss / i
    valAccuracy_AVG = valAccuracy / i
    return valLoss_AVG, valAccuracy_AVG


def trainCNN(trainDataloader, testDataloader):
    """
    Trains the Convolutional Neural Network using the following:
    Loss Function: Cross Entropy Loss function
    Optimizer: SGD with momentum

    :param trainDataloader: the Dataloader iterable of the trainDataset
    :param testDataloader: the Dataloader iterable of the testDataset
    :return:
    """
    print("[INFO] Training CNN...")
    lossFunction = nn.CrossEntropyLoss()
    modelOptimizer = optimizer.SGD(params = cnn.parameters(), lr = LR, momentum = 0.9)
    for epoch in range(EPOCHS):
        trainingLoss = 0.0
        for i, data in enumerate(trainDataloader):
            torch.cuda.empty_cache()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            modelOptimizer.zero_grad()
            outputs = cnn(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            modelOptimizer.step()
            trainingLoss += loss.item()
        trainingLoss_AVG = trainingLoss / float(i)
        validationLoss_AVG, validationAccuracy_AVG = accuracyCalculations(testDataloader, lossFunction)
        print("EPOCH:", epoch + 1)
        print("TRAINING LOSS =", trainingLoss_AVG)
        print("VALIDATION LOSS =", validationLoss_AVG)
        print("VALIDATION ACCURACY =", validationAccuracy_AVG, "%")
        epoch_data.append(epoch + 1)
        trainingLoss_data.append(trainingLoss_AVG)
        validationLoss_data.append(validationLoss_AVG)
        validationAccuracy_data.append(validationAccuracy_AVG)
        trainingLoss = 0.0

    print("...Done")
    saveModel()


def loadDataset(trainMean, trainSTD, testMean, testSTD):
    """
    Loads the CIFAR-10 Dataset and creates a dataloader for the train and test datasets

    :param trainMean: stores the calculated value of the Mean of the trainDataset
    :param trainSTD: stores the calculated value of the Standard Deviation of the trainDataset
    :param testMean: stores the calculated value of the Mean of the testDataset
    :param testSTD: stores the calculated value of the Standard Deviation of the testDataset
    :return:
    trainDataloader: the Dataloader iterable of the trainDataset
    testDataloader: the Dataloader iterable of the testDataset
    """
    print("[INFO] Loading Data from CIFAR 10...")
    trainTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    testTransform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

    trainDataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = trainTransform,
                                                download = True)
    testDataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = testTransform,
                                               download = False)

    trainDataloader = DataLoader(dataset = trainDataset, batch_size = BS, shuffle = True, num_workers = 0)
    testDataloader = DataLoader(dataset = testDataset, batch_size = BS, shuffle = False, num_workers = 0)

    print("trainDataloader = ", trainDataloader)
    print("testDataloader = ", testDataloader)

    print("...Done")

    return trainDataloader, testDataloader


def transformCalculations():
    """
    For calculating the mean and standard deviation values of the test and train datasets
    :return:
    trainMean: stores the calculated value of the Mean of the trainDataset
    trainSTD: stores the calculated value of the Standard Deviation of the trainDataset
    testMean: stores the calculated value of the Mean of the testDataset
    testSTD: stores the calculated value of the Standard Deviation of the testDataset
    """
    print("[INFO] Performing Transform Calculations...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainDataset = torchvision.datasets.CIFAR10(root = "./data", train = True, transform = transform,
                                                download = True)
    testDataset = torchvision.datasets.CIFAR10(root = "./data", train = False, transform = transform,
                                               download = True)

    # Calculating the Mean from the Train and Test Datasets
    trainMean = torch.from_numpy(trainDataset.data.mean(axis = (0, 1, 2)) / 255)
    trainMean = trainMean.to(device)
    trainSTD = torch.from_numpy(trainDataset.data.std(axis = (0, 1, 2)) / 255)
    trainSTD = trainSTD.to(device)
    testMean = torch.from_numpy(testDataset.data.mean(axis = (0, 1, 2)) / 255)
    testMean = testMean.to(device)
    testSTD = torch.from_numpy(testDataset.data.std(axis = (0, 1, 2)) / 255)
    testSTD = testSTD.to(device)

    print("trainMean:", trainMean)
    print("trainSTD:", trainSTD)
    print("testMean:", testMean)
    print("testSTD:", testSTD)

    print("...Done")

    return trainMean, trainSTD, testMean, testSTD


def main():
    """
    Main Function of the Program to call out other functions
    """
    trainMean, trainSTD, testMean, testSTD = transformCalculations()
    trainDataloader, testDataloader = loadDataset(trainMean, trainSTD, testMean, testSTD)
    trainCNN(trainDataloader, testDataloader)
    loadCNN()
    modelAccuracy(testDataloader)
    modelPlot()


if __name__ == "__main__":
    main()
