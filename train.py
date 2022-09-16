## executed command
## python train.py flowers --gpu

import argparse
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import json
import time


# Code hints from documentation:
# https://docs.python.org/3/library/argparse.html
# https://docs.python.org/3/library/argparse.html#the-add-argument-method
# Code hints from "Intro to Python. Lesson 6: Classifying Images"

'''
Train a new network on a data set with train.py

Basic usage: python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
'''
def main():

    def get_input_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('data_dir', action='store', help='The directory of the data')
        parser.add_argument('--save_dir', type=str, dest='save_dir', default='',
                            help='The directory of the checkpoints')
        parser.add_argument('--arch', type=str, dest='arch', default='vgg13', help='The chosen Model Architecture')
        parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.01, help='Learning Rate')
        parser.add_argument('--hidden_units', type=int, dest='hidden_units', default=512, help='Hidden units')
        parser.add_argument('--epochs', type=int, dest='epochs', default=20, help='Training Epochs')
        parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU')

        return parser.parse_args()

    # Define get_input_args() function to create & retrieve command line arguments
    in_args = get_input_args()

    print(
          "Command Line Arguments:" 
          "\n data_dir = ", in_args.data_dir,
          "\n save_dir = ", in_args.save_dir,
          "\n arch = ", in_args.arch,
          "\n learning_rate =", in_args.learning_rate,
          "\n hidden_units =", in_args.hidden_units,
          "\n epochs =", in_args.epochs,
          "\n gpu =", in_args.gpu,
        )

    # Model Architecture and build ## Hardware definitions #

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'

    # Transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Datasets
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # Model
    model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Classifier architechture
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replace Classifier
    model.classifier = classifier

    # Support Architechture
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Train and Validate

    def train_and_validate():
        print('Training Started. Please stand by...')
        epochs = 5
        print_every = 40
        steps = 0
        tt = time.time() 

        model.to(device)
        model.train()

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)      
                optimizer.zero_grad()       

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()       
                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    accuracy = 0
                    test_loss  = 0
                    for ii, (inputs, labels) in enumerate(testloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        test_loss  += criterion(output, labels)
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}".format(running_loss / print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss  / len(testloader)),
                          "Test Accuracy: {:.3f}%".format(accuracy / len(testloader)))

                    running_loss = 0
                    model.train()
        
        print("Training & Testing Finished in %s seconds." % (time.time() - tt)) 
        print("Time in minutes: ", ((time.time() - tt)) / 60)
        
        
        print('Validating accurancy. Please wait...')

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in validloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.train()
        print('The accuracy on the validation set is: %d %%' % (100 * correct / total))

    train_and_validate()

    #Save Checkpoint

    model.class_to_idx = train_data.class_to_idx

    arch =       {'input_size': 25088,
                  'output_size': 102,
                  'learning_rate': 0.001,
                  'epochs': 5,
                  'batch_size': 64,
                 }

    checkpoint = {'arch': arch,
                  'criterion': criterion,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                 }

    torch.save(checkpoint, 'checkpoint.pth')

    print("Your model has been saved!")

if __name__ == '__main__':
    main()


