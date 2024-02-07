import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import random
import sys
sys.path.append('../')  # Add parent directory to sys.path to access the 'src' directory
from src.vgg_local import VGG_LOCAL

def test(model: nn.Module, 
         criterion: nn.Module, 
         loader: DataLoader, 
         device: torch.device) -> None:
    """
    Test the model's performance on a dataset.

    Parameters:
    - model (torch.nn.Module): The model to be tested.
    - criterion (torch.nn.Module): The loss function used for testing.
    - loader (torch.utils.data.DataLoader): DataLoader for the dataset to be tested on.
    - device (torch.device): The device to which tensors should be moved before computation.
    """
    
    # Set the model to evaluation mode. In this mode, operations like dropout are disabled.
    model.eval()

    test_loss = 0  # Accumulated test loss
    correct = 0    # Count of correctly predicted samples
    total = 0      # Total samples processed

    # Disable gradient computations, as we are in evaluation mode and don't need gradients
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Test_Loss: {test_loss/len(loader)} | Test_Accuracy: {100.*correct/total}')

# Only test
if __name__ == '__main__':
    image_size = 64

    # Set random seed for reproducibility
    random_seed = 9999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    # Preprocessing for test data
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Prepare test data loader
    test_dataset = datasets.ImageFolder(root='../test_data_same', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize and load model weights
    model = VGG_LOCAL('VGG16', classes=3, image_size=image_size).to(device)
    model.load_state_dict(torch.load('../final_weight.pth'))

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Test the model
    test(model, criterion, test_loader, device)
