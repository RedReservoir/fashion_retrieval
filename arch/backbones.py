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
    


class EfficientNetB3Backbone(torch.nn.Module):
    """
    Standard EfficientNet-B3 feature backbone module.
    """


    def __init__(
            self,
            batchnorm_momentum=None,
            batchnorm_track_runnning_stats=None
            ):
        
        super(EfficientNetB3Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
        net = torchvision.models.efficientnet_b3(weights=weights)
        
        self.features = net.features

        # BatchNorm mods

        batchnorm_layers_list = self._compute_batchnorm_layers_list()
            
        if batchnorm_momentum is not None:
            for layer in batchnorm_layers_list:
                layer.momentum = batchnorm_momentum
            
        if batchnorm_track_runnning_stats is not None:
            for layer in batchnorm_layers_list:
                layer.track_running_stats = batchnorm_track_runnning_stats

        # Other parameters

        self.out_shape = (1536, 10, 10)


    def forward(self, x):
        
        x = self.features(x)

        return x


    def _compute_batchnorm_layers_list(self):

        batchnorm_layers_list = []

        batchnorm_layers_list += [
            self.features[0][1]
        ]

        batchnorm_layers_list += [
            self.features[1][0].block[0][1],
            self.features[1][0].block[2][1],
            self.features[1][1].block[0][1],
            self.features[1][1].block[2][1]
        ]
    
        batchnorm_layers_list += [
            self.features[2][0].block[0][1],
            self.features[2][0].block[1][1],
            self.features[2][0].block[3][1],
            self.features[2][1].block[0][1],
            self.features[2][1].block[1][1],
            self.features[2][1].block[3][1],
            self.features[2][2].block[0][1],
            self.features[2][2].block[1][1],
            self.features[2][2].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[3][0].block[0][1],
            self.features[3][0].block[1][1],
            self.features[3][0].block[3][1],
            self.features[3][1].block[0][1],
            self.features[3][1].block[1][1],
            self.features[3][1].block[3][1],
            self.features[3][2].block[0][1],
            self.features[3][2].block[1][1],
            self.features[3][2].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[4][0].block[0][1],
            self.features[4][0].block[1][1],
            self.features[4][0].block[3][1],
            self.features[4][1].block[0][1],
            self.features[4][1].block[1][1],
            self.features[4][1].block[3][1],
            self.features[4][2].block[0][1],
            self.features[4][2].block[1][1],
            self.features[4][2].block[3][1],
            self.features[4][3].block[0][1],
            self.features[4][3].block[1][1],
            self.features[4][3].block[3][1],
            self.features[4][4].block[0][1],
            self.features[4][4].block[1][1],
            self.features[4][4].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[5][0].block[0][1],
            self.features[5][0].block[1][1],
            self.features[5][0].block[3][1],
            self.features[5][1].block[0][1],
            self.features[5][1].block[1][1],
            self.features[5][1].block[3][1],
            self.features[5][2].block[0][1],
            self.features[5][2].block[1][1],
            self.features[5][2].block[3][1],
            self.features[5][3].block[0][1],
            self.features[5][3].block[1][1],
            self.features[5][3].block[3][1],
            self.features[5][4].block[0][1],
            self.features[5][4].block[1][1],
            self.features[5][4].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[6][0].block[0][1],
            self.features[6][0].block[1][1],
            self.features[6][0].block[3][1],
            self.features[6][1].block[0][1],
            self.features[6][1].block[1][1],
            self.features[6][1].block[3][1],
            self.features[6][2].block[0][1],
            self.features[6][2].block[1][1],
            self.features[6][2].block[3][1],
            self.features[6][3].block[0][1],
            self.features[6][3].block[1][1],
            self.features[6][3].block[3][1],
            self.features[6][4].block[0][1],
            self.features[6][4].block[1][1],
            self.features[6][4].block[3][1],
            self.features[6][5].block[0][1],
            self.features[6][5].block[1][1],
            self.features[6][5].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[7][0].block[0][1],
            self.features[7][0].block[1][1],
            self.features[7][0].block[3][1],
            self.features[7][1].block[0][1],
            self.features[7][1].block[1][1],
            self.features[7][1].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[8][1]
        ]

        return batchnorm_layers_list
    

    
class EfficientNetB4Backbone(torch.nn.Module):
    """
    Standard EfficientNet-B4 feature backbone module.
    """


    def __init__(
            self,
            batchnorm_momentum=None,
            batchnorm_track_runnning_stats=None
            ):
                
        super(EfficientNetB4Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        net = torchvision.models.efficientnet_b4(weights=weights)
        
        self.features = net.features

        # BatchNorm mods

        batchnorm_layers_list = self._compute_batchnorm_layers_list()
            
        if batchnorm_momentum is not None:
            for layer in batchnorm_layers_list:
                layer.momentum = batchnorm_momentum
            
        if batchnorm_track_runnning_stats is not None:
            for layer in batchnorm_layers_list:
                layer.track_running_stats = batchnorm_track_runnning_stats

        # Other parameters

        self.out_shape = (1792, 12, 12)


    def forward(self, x):
        
        x = self.features(x)

        return x


    def _compute_batchnorm_layers_list(self):

        batchnorm_layers_list = []

        batchnorm_layers_list += [
            self.features[0][1]
        ]

        batchnorm_layers_list += [
            self.features[1][0].block[0][1],
            self.features[1][0].block[2][1],
            self.features[1][1].block[0][1],
            self.features[1][1].block[2][1]
        ]
    
        batchnorm_layers_list += [
            self.features[2][0].block[0][1],
            self.features[2][0].block[1][1],
            self.features[2][0].block[3][1],
            self.features[2][1].block[0][1],
            self.features[2][1].block[1][1],
            self.features[2][1].block[3][1],
            self.features[2][2].block[0][1],
            self.features[2][2].block[1][1],
            self.features[2][2].block[3][1],
            self.features[2][3].block[0][1],
            self.features[2][3].block[1][1],
            self.features[2][3].block[3][1]
        ]
    
        batchnorm_layers_list += [
            self.features[3][0].block[0][1],
            self.features[3][0].block[1][1],
            self.features[3][0].block[3][1],
            self.features[3][1].block[0][1],
            self.features[3][1].block[1][1],
            self.features[3][1].block[3][1],
            self.features[3][2].block[0][1],
            self.features[3][2].block[1][1],
            self.features[3][2].block[3][1],
            self.features[3][3].block[0][1],
            self.features[3][3].block[1][1],
            self.features[3][3].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[4][0].block[0][1],
            self.features[4][0].block[1][1],
            self.features[4][0].block[3][1],
            self.features[4][1].block[0][1],
            self.features[4][1].block[1][1],
            self.features[4][1].block[3][1],
            self.features[4][2].block[0][1],
            self.features[4][2].block[1][1],
            self.features[4][2].block[3][1],
            self.features[4][3].block[0][1],
            self.features[4][3].block[1][1],
            self.features[4][3].block[3][1],
            self.features[4][4].block[0][1],
            self.features[4][4].block[1][1],
            self.features[4][4].block[3][1],
            self.features[4][5].block[0][1],
            self.features[4][5].block[1][1],
            self.features[4][5].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[5][0].block[0][1],
            self.features[5][0].block[1][1],
            self.features[5][0].block[3][1],
            self.features[5][1].block[0][1],
            self.features[5][1].block[1][1],
            self.features[5][1].block[3][1],
            self.features[5][2].block[0][1],
            self.features[5][2].block[1][1],
            self.features[5][2].block[3][1],
            self.features[5][3].block[0][1],
            self.features[5][3].block[1][1],
            self.features[5][3].block[3][1],
            self.features[5][4].block[0][1],
            self.features[5][4].block[1][1],
            self.features[5][4].block[3][1],
            self.features[5][5].block[0][1],
            self.features[5][5].block[1][1],
            self.features[5][5].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[6][0].block[0][1],
            self.features[6][0].block[1][1],
            self.features[6][0].block[3][1],
            self.features[6][1].block[0][1],
            self.features[6][1].block[1][1],
            self.features[6][1].block[3][1],
            self.features[6][2].block[0][1],
            self.features[6][2].block[1][1],
            self.features[6][2].block[3][1],
            self.features[6][3].block[0][1],
            self.features[6][3].block[1][1],
            self.features[6][3].block[3][1],
            self.features[6][4].block[0][1],
            self.features[6][4].block[1][1],
            self.features[6][4].block[3][1],
            self.features[6][5].block[0][1],
            self.features[6][5].block[1][1],
            self.features[6][5].block[3][1],
            self.features[6][6].block[0][1],
            self.features[6][6].block[1][1],
            self.features[6][6].block[3][1],
            self.features[6][7].block[0][1],
            self.features[6][7].block[1][1],
            self.features[6][7].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[7][0].block[0][1],
            self.features[7][0].block[1][1],
            self.features[7][0].block[3][1],
            self.features[7][1].block[0][1],
            self.features[7][1].block[1][1],
            self.features[7][1].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[8][1]
        ]

        return batchnorm_layers_list



class EfficientNetB5Backbone(torch.nn.Module):
    """
    Standard EfficientNet-B5 feature backbone module.
    """


    def __init__(
            self,
            batchnorm_momentum=None,
            batchnorm_track_runnning_stats=None
            ):
        
        super(EfficientNetB5Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
        net = torchvision.models.efficientnet_b5(weights=weights)
        
        self.features = net.features

        # BatchNorm mods

        batchnorm_layers_list = self._compute_batchnorm_layers_list()
            
        if batchnorm_momentum is not None:
            for layer in batchnorm_layers_list:
                layer.momentum = batchnorm_momentum
            
        if batchnorm_track_runnning_stats is not None:
            for layer in batchnorm_layers_list:
                layer.track_running_stats = batchnorm_track_runnning_stats

        # Other parameters

        self.out_shape = (2048, 15, 15)


    def forward(self, x):
        
        x = self.features(x)

        return x


    def _compute_batchnorm_layers_list(self):

        batchnorm_layers_list = []

        batchnorm_layers_list += [
            self.features[0][1]
        ]

        batchnorm_layers_list += [
            self.features[1][0].block[0][1],
            self.features[1][0].block[2][1],
            self.features[1][1].block[0][1],
            self.features[1][1].block[2][1],
            self.features[1][2].block[0][1],
            self.features[1][2].block[2][1]
        ]
    
        batchnorm_layers_list += [
            self.features[2][0].block[0][1],
            self.features[2][0].block[1][1],
            self.features[2][0].block[3][1],
            self.features[2][1].block[0][1],
            self.features[2][1].block[1][1],
            self.features[2][1].block[3][1],
            self.features[2][2].block[0][1],
            self.features[2][2].block[1][1],
            self.features[2][2].block[3][1],
            self.features[2][3].block[0][1],
            self.features[2][3].block[1][1],
            self.features[2][3].block[3][1],
            self.features[2][4].block[0][1],
            self.features[2][4].block[1][1],
            self.features[2][4].block[3][1]
        ]
    
        batchnorm_layers_list += [
            self.features[3][0].block[0][1],
            self.features[3][0].block[1][1],
            self.features[3][0].block[3][1],
            self.features[3][1].block[0][1],
            self.features[3][1].block[1][1],
            self.features[3][1].block[3][1],
            self.features[3][2].block[0][1],
            self.features[3][2].block[1][1],
            self.features[3][2].block[3][1],
            self.features[3][3].block[0][1],
            self.features[3][3].block[1][1],
            self.features[3][3].block[3][1],
            self.features[3][4].block[0][1],
            self.features[3][4].block[1][1],
            self.features[3][4].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[4][0].block[0][1],
            self.features[4][0].block[1][1],
            self.features[4][0].block[3][1],
            self.features[4][1].block[0][1],
            self.features[4][1].block[1][1],
            self.features[4][1].block[3][1],
            self.features[4][2].block[0][1],
            self.features[4][2].block[1][1],
            self.features[4][2].block[3][1],
            self.features[4][3].block[0][1],
            self.features[4][3].block[1][1],
            self.features[4][3].block[3][1],
            self.features[4][4].block[0][1],
            self.features[4][4].block[1][1],
            self.features[4][4].block[3][1],
            self.features[4][5].block[0][1],
            self.features[4][5].block[1][1],
            self.features[4][5].block[3][1],
            self.features[4][6].block[0][1],
            self.features[4][6].block[1][1],
            self.features[4][6].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[5][0].block[0][1],
            self.features[5][0].block[1][1],
            self.features[5][0].block[3][1],
            self.features[5][1].block[0][1],
            self.features[5][1].block[1][1],
            self.features[5][1].block[3][1],
            self.features[5][2].block[0][1],
            self.features[5][2].block[1][1],
            self.features[5][2].block[3][1],
            self.features[5][3].block[0][1],
            self.features[5][3].block[1][1],
            self.features[5][3].block[3][1],
            self.features[5][4].block[0][1],
            self.features[5][4].block[1][1],
            self.features[5][4].block[3][1],
            self.features[5][5].block[0][1],
            self.features[5][5].block[1][1],
            self.features[5][5].block[3][1],
            self.features[5][6].block[0][1],
            self.features[5][6].block[1][1],
            self.features[5][6].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[6][0].block[0][1],
            self.features[6][0].block[1][1],
            self.features[6][0].block[3][1],
            self.features[6][1].block[0][1],
            self.features[6][1].block[1][1],
            self.features[6][1].block[3][1],
            self.features[6][2].block[0][1],
            self.features[6][2].block[1][1],
            self.features[6][2].block[3][1],
            self.features[6][3].block[0][1],
            self.features[6][3].block[1][1],
            self.features[6][3].block[3][1],
            self.features[6][4].block[0][1],
            self.features[6][4].block[1][1],
            self.features[6][4].block[3][1],
            self.features[6][5].block[0][1],
            self.features[6][5].block[1][1],
            self.features[6][5].block[3][1],
            self.features[6][6].block[0][1],
            self.features[6][6].block[1][1],
            self.features[6][6].block[3][1],
            self.features[6][7].block[0][1],
            self.features[6][7].block[1][1],
            self.features[6][7].block[3][1],
            self.features[6][8].block[0][1],
            self.features[6][8].block[1][1],
            self.features[6][8].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[7][0].block[0][1],
            self.features[7][0].block[1][1],
            self.features[7][0].block[3][1],
            self.features[7][1].block[0][1],
            self.features[7][1].block[1][1],
            self.features[7][1].block[3][1],
            self.features[7][2].block[0][1],
            self.features[7][2].block[1][1],
            self.features[7][2].block[3][1]
        ]

        batchnorm_layers_list += [
            self.features[8][1]
        ]

        return batchnorm_layers_list



class ConvNeXtTinyBackbone(torch.nn.Module):
    """
    Standard ConvNeXt Tiny feature backbone module.
    """


    def __init__(self):
        
        super(ConvNeXtTinyBackbone, self).__init__()

        # Model construction

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        net = torchvision.models.convnext_tiny(weights=weights)
        
        self.features = net.features

        # Other parameters

        self.out_shape = (768, 7, 7)


    def forward(self, x):
        
        x = self.features(x)

        return x
