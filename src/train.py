from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch

def train(epoch: int, 
          model: Module, 
          optimizer: Optimizer, 
          criterion: Module, 
          loader: DataLoader, 
          device: torch.device):
    """
    Train the model for one epoch on a given dataset.

    Parameters:
    - epoch (int): Current epoch number.
    - model (torch.nn.Module): The model to be trained.
    - optimizer (torch.optim.Optimizer): The optimizer for model training.
    - criterion (torch.nn.Module): The loss function used for training.
    - loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - device (torch.device): The device to which tensors should be moved before computation.
    """
    
    # Set the model to training mode. This enables operations which are only active during training like dropout.
    model.train()

    # Initialize counters and accumulators
    train_loss = 0  # Accumulated training loss
    correct = 0     # Count of correctly predicted samples
    total = 0       # Total samples processed

    # Display a progress bar using tqdm for the batches
    progress_bar = tqdm(loader)
    for (inputs, targets) in progress_bar:
        # Move data and target tensors to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the optimizer's gradient buffers to not accumulate gradients across batches
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(inputs)
        
        # Compute the loss between predictions and ground truth
        loss = criterion(outputs, targets)
        
        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate loss and update counters
        train_loss += loss.item()
        _, predicted = outputs.max(1)  # Get the index of the max log-probability as prediction
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Print training results for the epoch
    print(f'Epoch: {epoch} | Train_Loss: {train_loss/len(loader)} | Train_Accuracy: {100.*correct/total}')

    return train_loss/len(loader), correct/total