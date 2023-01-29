import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset
import cv2
from sklearn.preprocessing import LabelBinarizer

BS = 128
EPOCHS = 5
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
            nn.Linear(1024, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 26),
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
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            images = images.permute(0, 3, 1, 2)
            outputs = cnn(images)
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
        labels = labels.type(torch.LongTensor)
        images, labels = images.to(device), labels.to(device)
        images = images.permute(0, 3, 1, 2)
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
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            modelOptimizer.zero_grad()
            inputs = inputs.permute(0, 3, 1, 2)
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

def load_az_dataset(datasetPath):
	# initialize the list of data and labels
	data = []
	labels = []
	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):
		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))
		# update the list of data and labels
		data.append(image)
		labels.append(label)
	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")
	# return a 2-tuple of the A-Z data and labels
	return (data, labels)

def loadDataset():
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

    (azData, azLabels) = load_az_dataset("C:/Users/karan/Software Projects/WARG_CV_Bootcamp/A_Z Handwritten Data.csv")
    data = np.vstack([azData])
    labels = np.hstack([azLabels])
    print(data)
    print(labels)
    data = [cv2.resize(image, (32, 32)) for image in data]
    #data = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in data]
    data = np.array(data, dtype="float32")
    #data = data.reshape(1, 32, 32)
    #labels = labels.reshape(labels.shape[0])
    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    
    trainDataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
    testDataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY))


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
    (azData, azLabels) = load_az_dataset("C:/Users/karan/Software Projects/WARG_CV_Bootcamp/A_Z Handwritten Data.csv")
    data = np.vstack([azData])
    labels = np.hstack([azLabels])
    data = [cv2.resize(image, (32, 32)) for image in data]

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    
    trainDataset = TensorDataset(torch.from_numpy(trainX).to(torch.float32), torch.from_numpy(trainY).to(torch.float32))
    testDataset = TensorDataset(torch.from_numpy(testX).to(torch.float32), torch.from_numpy(testY).to(torch.float32))

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

#data.head()
class traindataset(Dataset):
    def __init__(self,data,train_end_idx,augmentation = None):
        '''
        data: pandas dataframe
        
        '''
        self.data=data
        self.augmentation=augmentation
        self.train_end=train_end_idx
        self.target=self.data.iloc[:self.train_end,1].values
        self.image=self.data.iloc[:self.train_end,2:].values#contains full data
        
    def __len__(self):
        return len(self.target);
    def __getitem__(self,idx):
        
        self.target=self.target
        self.ima=self.image[idx].reshape(28,28) #only takes the selected index
        if self.augmentation is not None:
            self.ima = self.augmentation(self.ima)
        
        return torch.tensor(self.target[idx]),torch.tensor(self.ima)
    
class valdataset(Dataset):
    def __init__(self,data,val_start_idx,val_end_idx,augmentation=None):
        self.data=data
        self.augmentation=augmentation
        self.val_start_idx=val_start_idx
        self.val_end_idx=val_end_idx
        self.target=self.data.iloc[self.val_start_idx:self.val_end_idx,1].values
        self.image=self.data.iloc[self.val_start_idx:self.val_end_idx,2:].values
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self,idx):
        target=self.target
        self.ima=self.image[idx].reshape(28,28)
        if self.augmentation is not None:
            self.ima = self.augmentation(self.ima)
            #print(self.ima)
        #return torch.tensor(target[idx]),torch.from_numpy(self.ima)
        return torch.tensor(target[idx]),torch.tensor(self.ima)

def readCSV():
    data = pd.read_csv("TMNIST_Data.csv", nrows= 10000)
    le=preprocessing.LabelEncoder()
    data.labels=le.fit_transform(data.labels)
    torchvision_transform = transforms.Compose([
        np.uint8,
        transforms.ToPILImage(),
        #transforms.Resize((28,28)),
        # #transforms.RandomRotation([45,135]),
        # #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    trainds=traindataset(data,1000,torchvision_transform)
    train_loader=DataLoader(trainds,batch_size=8,shuffle=True)
    valds=valdataset(data,1001,1300,torchvision_transform)
    val_loader=DataLoader(valds,batch_size=8,shuffle=True)

    return train_loader, val_loader

def load_az_dataset(datasetPath):
	# initialize the list of data and labels
	data = []
	labels = []
	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):
		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))
		# update the list of data and labels
		data.append(image)
		labels.append(label)
	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")
	# return a 2-tuple of the A-Z data and labels
	return (data, labels)

def newfn():
    (azData, azLabels) = load_az_dataset("A_Z Handwritten Data.csv")
    data = np.vstack([azData])
    labels = np.hstack([azLabels])

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    
    trainDataset = TensorDataset(torch.from_numpy(trainX).to(torch.float32), torch.from_numpy(trainY).to(torch.float32))
    trainDataset = TensorDataset(torch.from_numpy(trainX).to(torch.float32), torch.from_numpy(trainY).to(torch.float32))


    #train, rest = train_test_split(data, test_size=0.4, random_state=42)
    #x_train, x_test, y_train, y_test = train_test_split(rest[:,1:],rest[:,0],test_size=0.4)
    #lb = LabelBinarizer()
    y_train_label = lb.fit_transform(y_train)
    y_test_label = lb.transform(y_test)
    x_train = (x_train.values / 255).reshape(-1, 1, 28, 28)
    x_test = (x_test.values / 255).reshape(-1, 1, 28, 28)
    train_set = TensorDataset(torch.from_numpy(x_train).to(torch.float32),
                          torch.from_numpy(y_train_label).to(torch.float32))
    test_set = TensorDataset(torch.from_numpy(x_test).to(torch.float32), torch.from_numpy(y_test_label).to(torch.float32))
    train_loader = DataLoader(train_set, batch_size=92, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=92)
    return train_loader, test_loader

def main():
    """
    Main Function of the Program to call out other functions
    """
    #trainDataloader, testDataloader = readCSV()
    #trainDataloader, testDataloader = newfn()
    #trainMean, trainSTD, testMean, testSTD = transformCalculations()
    trainDataloader, testDataloader = loadDataset()
    trainCNN(trainDataloader, testDataloader)
    loadCNN()
    modelAccuracy(testDataloader)
    modelPlot()


if __name__ == "__main__":
    main()
