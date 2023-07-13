import torch
import torchvision



class ResNet50Backbone(torch.nn.Module):


    def __init__(self):
        
        super(ResNet50Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.ResNet50_Weights.DEFAULT
        resnet = torchvision.models.resnet50(weights=weights)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

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
