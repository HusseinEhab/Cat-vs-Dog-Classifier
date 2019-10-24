import torch
import torchvision
from torch import nn, optim
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import cv2


def validation(model, testloader, criterion, device="cpu"):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss, accuracy


def training(model, trainloader, validloader, criterion, optimizer, device="gpu", epochs=10):
    valid_losses = []
    best_acc = 0
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:

            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            model.eval()
            valid_loss = 0
            valid_accuracy = 0
            with torch.no_grad():
                valid_loss, valid_accuracy = validation(model, validloader, criterion, device)
                valid_losses.append(valid_loss / len(validloader))
            print("epoch {0}".format(e))
            print("Training loss = {0} ".format(running_loss / len(trainloader)))
            print("validation loss = {0} ".format(valid_loss / len(validloader)))
            print("Test accuracy = {0} % \n".format((valid_accuracy / len(validloader)) * 100))
            if best_acc < ((valid_accuracy / len(validloader)) * 100):
                best_acc = ((valid_accuracy / len(validloader)) * 100)
                torch.save(model.state_dict(), "checkpoint.pth")

    return valid_loss


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

dataset = torchvision.datasets.ImageFolder("C:/train", transform=train_transforms)

train_set, Validation_set = torch.utils.data.random_split(dataset, (20000, 2500))
trainloader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
validloader = torch.utils.data.DataLoader(Validation_set, batch_size=256, shuffle=True)

test_data = datasets.ImageFolder('C:/test', transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# defining Network, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)
print("Device {0}".format(device))

valid_loss = training(model, trainloader, validloader, criterion, optimizer, device, 30)
loss_test, acc_test = validation(model, testloader, criterion, device)

plt.plot(valid_loss, label='valid_loss')
plt.xlabel("epoch")
plt.ylabel("Validation Loss ")
plt.legend(frameon=False)
plt.show()

print((acc_test / len(testloader)) * 100)

