'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from typing import List, Tuple, Union

# Configuration for different VGG variants.
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_LOCAL(nn.Module):
    def __init__(self, model: str, classes: int, image_size: int) -> None:
        super(VGG_LOCAL, self).__init__()
        self.features = self._make_layers(cfg[model])
        
        # Compute flattened size after feature extraction
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

    def _make_layers(self, cfg: List[Union[int, str]]) -> nn.Sequential:
        """
        Generate layers based on the given configuration.
        """
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, last: bool = False, freeze: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.

        Parameters:
        - x: input tensor.
        - last: if True, returns the output before the classifier.
        - freeze: if True, the features extraction is performed with torch.no_grad() for faster computation.
        """
        if freeze:
            with torch.no_grad():
                out = self.features(x)
                e = out.view(out.size(0), -1)
        else:
            out = self.features(x)
            e = out.view(out.size(0), -1)
        out = self.classifier(e)
        if last:
            return out, e
        else:
            return out
