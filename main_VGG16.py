import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import random

from src.vgg import VGG
from src.train import train
from src.eval import eval
from src.test import test

def main(model: str = 'VGG16',      # Name of the model to use
         classes: int = 10,          # Number of output dimensions (classes)
         image_size: int = 32,      # Size of the input image (width and height)
         pretrained: bool = False    # Whether to use a pre-trained model
        ) -> None:
    """
    Initializes and sets up the neural network model.
    
    :model(str): The name of the model to use.
    :classes(int): The number of classes to classify.
    :image_size(int): The size of the images to input into the network.
    :pretrained(bool): Flag indicating whether to use a pre-trained model.
    """

    # Fixing the seed for reproducibility
    random_seed = 9999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # Checking if GPU is available and setting the seed for GPU operations
    device = torch.device('mps')
        

    # Preprocess and augmentation for training data
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Preprocess for test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Loading and splitting the dataset into train, validation, and test sets
    # HACK janken dataset
    # dataset = datasets.ImageFolder(root='./train_data', transform=test_transform)
    # test_dataset = datasets.ImageFolder(root='./test_data_same', transform=test_transform)
    # HACK CIFAR10 dataset
    dataset = datasets.CIFAR10(root='./cifar', train=True, transform=test_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./cifar', train=False, transform=test_transform, download=True)
    train_len = int(len(dataset)*0.9)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_dataset.transform = train_transform

    # Preparing data loaders for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Creating the model, loss function, and optimizer
    model = VGG(model, classes=classes, image_size=image_size, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Initialize result lists
    train_result = []; eval_result = []

    # Training loop
    for epoch in range(1, 11):
        train_loss, train_acc = train(epoch, model, optimizer, criterion, train_loader, device)
        eval_loss, eval_acc = eval(epoch, model, criterion, val_loader, device)
        train_result.append((train_acc, train_loss))
        eval_result.append((eval_acc, eval_loss))
    print('###############Training Done###############')

    # Saving the results
    train_acc, train_loss = zip(*train_result)
    eval_acc, eval_loss = zip(*eval_result)
    fig = plt.figure()
    plt.plot(train_acc, label='train_acc')
    plt.plot(eval_acc, label='eval_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    fig.savefig('acc.png')

    fig = plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(eval_loss, label='eval_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('loss.png')

    # Testing the model after training
    test(model, criterion, test_loader, device)

    # Saving the trained model weights
    torch.save(model.state_dict(), f'./final_weight.pth')
    print(f'Saved model to ./final_weight.pth')

if __name__ == '__main__':
    main()
