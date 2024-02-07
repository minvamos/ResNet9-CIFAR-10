import torch
from torch.nn import Module
from torch.utils.data import DataLoader

def eval(epoch: int, 
         model: Module, 
         criterion: Module, 
         loader: DataLoader, 
         device: torch.device):
    """
    Evaluate the model's performance on a dataset.

    Parameters:
    - epoch (int): Current epoch number.
    - model (torch.nn.Module): The model to be evaluated.
    - criterion (torch.nn.Module): The loss function used for evaluation.
    - loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated on.
    - device (torch.device): The device to which tensors should be moved before computation.
    """
    
    # Set the model to evaluation mode. In this mode, certain operations like dropout are disabled.
    model.eval()
    
    eval_loss = 0  # Accumulated evaluation loss
    correct = 0    # Count of correctly predicted samples
    total = 0      # Total samples processed
    
    # Disable gradient computations. Since we're in evaluation mode, we don't need gradients.
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Epoch: {epoch} | Eval_Loss: {eval_loss/len(loader)} | Eval_Accuracy: {100.*correct/total}')

    return eval_loss/len(loader), correct/total
