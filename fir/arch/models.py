import torch
import torchvision

from arch import heads



class BackboneAndHead(torch.nn.Module):
    """
    Basic backbone and head model.
    """


    def __init__(self, backbone, head):
        
        super(BackboneAndHead, self).__init__()

        self.backbone = backbone
        self.head = head


    def forward(self, x):
        
        x = self.backbone(x)
        x = self.head(x)

        return x
