import torch
import torchvision



class ResNet50Backbone(torch.nn.Module):
    """
    Standard ResNet50 feature backbone module.
    """

    def __init__(self):
        
        super(ResNet50Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.ResNet50_Weights.DEFAULT
        net = torchvision.models.resnet50(weights=weights)
        
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Other parameters

        self.out_shape = (2048, 7, 7)


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    


class EfficientNetB5Backbone(torch.nn.Module):
    """
    Standard EfficientNet B5 feature backbone module.
    """


    def __init__(self):
        
        super(EfficientNetB5Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
        net = torchvision.models.efficientnet_b5(weights=weights)
        
        self.features = net.features

        # Other parameters

        self.out_shape = (2048, 15, 15)


    def forward(self, x):
        
        x = self.features(x)

        return x



class ConvNeXTTinyBackbone(torch.nn.Module):
    """
    Standard ConvNeXT Tiny feature backbone module.
    """


    def __init__(self):
        
        super(ConvNeXTTinyBackbone, self).__init__()

        # Model construction

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        net = torchvision.models.convnext_tiny(weights=weights)
        
        self.features = net.features

        # Other parameters

        self.out_shape = (768, 7, 7)


    def forward(self, x):
        
        x = self.features(x)

        return x
