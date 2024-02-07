import torch
import torch.nn as nn
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

class VGG(nn.Module):
    def __init__(self, model: str, classes: int, image_size: int = 224, pretrained=False) -> None:
        """
        VGG model constructor.

        Parameters:
        - model (str): The type of VGG model ('VGG11', 'VGG13', 'VGG16', or 'VGG19').
        - classes (int): The number of output classes.
        - image_size (int, optional): The height (and width) of input images. Default is 224.
        """
        super(VGG, self).__init__()
        
        # Use appropriate VGG model based on the provided model type
        if model == 'VGG11':
            self.features = vgg11_bn(pretrained=pretrained).features
        elif model == 'VGG13':
            self.features = vgg13_bn(pretrained=pretrained).features
        elif model == 'VGG16':
            self.features = vgg16_bn(pretrained=pretrained).features
        elif model == 'VGG19':
            self.features = vgg19_bn(pretrained=pretrained).features
        else:
            raise ValueError('Unsupported VGG model')
        
        # Compute the size of the flattened features to be fed to the classifier
        with torch.no_grad():
            self.flattened_size = self.features(torch.zeros(1, 3, image_size, image_size)).view(-1).shape[0]
        
        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VGG model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Model's output tensor.
        """
        x = self.features(x)             # Apply the feature extractor layers
        x = x.view(x.size(0), -1)       # Flatten the output
        x = self.classifier(x)          # Apply the classifier layers
        return x
