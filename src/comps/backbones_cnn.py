import torch
import torchvision



class Contiguous(torch.nn.Module):
    """
    Non-trainable module that applies torch.contiguous() to a tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous()



class LayerNorm2dContiguous(torch.nn.LayerNorm):
    """
    Version of pytorch/vision/torchvision/models/convnext/LayerNorm2d with torch.contiguous()
    at the end of the forward pass. 
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        return x



########



class ResNet50Backbone(torch.nn.Module):
    """
    Standard ResNet50 feature backbone module.
    Code interface from PyTorch.
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

        # Feature shape

        self.feature_shape = (2048, 7, 7)


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
    

    def get_image_transform(self):

        image_transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform
    


class EfficientNetB2Backbone(torch.nn.Module):
    """
    Standard EfficientNet-B2 feature backbone module.
    Code interface from PyTorch.
    """


    def __init__(
            self,
            batchnorm_momentum=None,
            batchnorm_track_runnning_stats=None,
            batchnorm_eps=None,
            silu_inplace=None
            ):
        
        super(EfficientNetB2Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        net = torchvision.models.efficientnet_b2(weights=weights)
        
        self.features = net.features

        # BatchNorm mods

        batchnorm_layers_list = self._get_batchnorm_layers()
            
        if batchnorm_momentum is not None:
            for layer in batchnorm_layers_list:
                layer.momentum = batchnorm_momentum
            
        if batchnorm_track_runnning_stats is not None:
            for layer in batchnorm_layers_list:
                layer.track_running_stats = batchnorm_track_runnning_stats
            
        if batchnorm_eps is not None:
            for layer in batchnorm_layers_list:
                layer.eps = batchnorm_eps

        # Inplace mods

        silu_layers_list = self._get_silu_layers()

        if silu_inplace is not None:
            for layer in silu_layers_list:
                layer.inplace = silu_inplace

        # Feature shape

        self.feature_shape = (1408, 9, 9)


    def forward(self, x):
        
        x = self.features(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform


    def _get_batchnorm_layers(self):

        batchnorm_layers_list = []

        batchnorm_layers_list += [
            self.features[0][1]
        ]

        batchnorm_layers_list += [
            self.features[1][idx_1].block[idx_2][1] for idx_1 in range(2) for idx_2 in [0, 2]
        ]

        batchnorm_layers_list += [
            self.features[2][idx_1].block[idx_2][1] for idx_1 in range(3) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[3][idx_1].block[idx_2][1] for idx_1 in range(3) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[4][idx_1].block[idx_2][1] for idx_1 in range(4) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[5][idx_1].block[idx_2][1] for idx_1 in range(4) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[6][idx_1].block[idx_2][1] for idx_1 in range(5) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[7][idx_1].block[idx_2][1] for idx_1 in range(2) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[8][1]
        ]

        return batchnorm_layers_list
    

    def _get_silu_layers(self):

        silu_layers_list = []

        silu_layers_list += [
            self.features[0][2]
        ]

        silu_layers_list += [
            self.features[1][idx].block[0][2] for idx in range(2)
        ]

        silu_layers_list += [
            self.features[1][idx].block[1].activation for idx in range(2)
        ]

        silu_layers_list += [
            self.features[2][idx_1].block[idx_2][2] for idx_1 in range(3) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[2][idx].block[2].activation for idx in range(3)
        ]

        silu_layers_list += [
            self.features[3][idx_1].block[idx_2][2] for idx_1 in range(3) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[3][idx].block[2].activation for idx in range(3)
        ]

        silu_layers_list += [
            self.features[4][idx_1].block[idx_2][2] for idx_1 in range(4) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[4][idx].block[2].activation for idx in range(4)
        ]

        silu_layers_list += [
            self.features[5][idx_1].block[idx_2][2] for idx_1 in range(4) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[5][idx].block[2].activation for idx in range(4)
        ]

        silu_layers_list += [
            self.features[6][idx_1].block[idx_2][2] for idx_1 in range(5) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[6][idx].block[2].activation for idx in range(5)
        ]

        silu_layers_list += [
            self.features[7][idx_1].block[idx_2][2] for idx_1 in range(2) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[7][idx].block[2].activation for idx in range(2)
        ]

        silu_layers_list += [
            self.features[8][2]
        ]

        return silu_layers_list



class EfficientNetB3Backbone(torch.nn.Module):
    """
    Standard EfficientNet-B3 feature backbone module.
    Code interface from PyTorch.
    """


    def __init__(
            self,
            batchnorm_momentum=None,
            batchnorm_track_runnning_stats=None,
            batchnorm_eps=None,
            silu_inplace=None
            ):
        
        super(EfficientNetB3Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
        net = torchvision.models.efficientnet_b3(weights=weights)
        
        self.features = net.features

        # BatchNorm mods

        batchnorm_layers_list = self._get_batchnorm_layers()
            
        if batchnorm_momentum is not None:
            for layer in batchnorm_layers_list:
                layer.momentum = batchnorm_momentum
            
        if batchnorm_track_runnning_stats is not None:
            for layer in batchnorm_layers_list:
                layer.track_running_stats = batchnorm_track_runnning_stats
            
        if batchnorm_eps is not None:
            for layer in batchnorm_layers_list:
                layer.eps = batchnorm_eps

        # Inplace mods

        silu_layers_list = self._get_silu_layers()

        if silu_inplace is not None:
            for layer in silu_layers_list:
                layer.inplace = silu_inplace

        # Feature shape

        self.feature_shape = (1536, 10, 10)


    def forward(self, x):
        
        x = self.features(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform


    def _get_batchnorm_layers(self):

        batchnorm_layers_list = []

        batchnorm_layers_list += [
            self.features[0][1]
        ]
    
        batchnorm_layers_list += [
            self.features[1][idx_1].block[idx_2][1] for idx_1 in range(2) for idx_2 in [0, 2]
        ]

        batchnorm_layers_list += [
            self.features[2][idx_1].block[idx_2][1] for idx_1 in range(3) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[3][idx_1].block[idx_2][1] for idx_1 in range(3) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[4][idx_1].block[idx_2][1] for idx_1 in range(5) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[5][idx_1].block[idx_2][1] for idx_1 in range(5) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[6][idx_1].block[idx_2][1] for idx_1 in range(6) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[7][idx_1].block[idx_2][1] for idx_1 in range(2) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[8][1]
        ]

        return batchnorm_layers_list
    

    def _get_silu_layers(self):

        silu_layers_list = []

        silu_layers_list += [
            self.features[0][2]
        ]

        silu_layers_list += [
            self.features[1][idx].block[0][2] for idx in range(2)
        ]

        silu_layers_list += [
            self.features[1][idx].block[1].activation for idx in range(2)
        ]

        silu_layers_list += [
            self.features[2][idx_1].block[idx_2][2] for idx_1 in range(3) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[2][idx].block[2].activation for idx in range(3)
        ]

        silu_layers_list += [
            self.features[3][idx_1].block[idx_2][2] for idx_1 in range(3) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[3][idx].block[2].activation for idx in range(3)
        ]

        silu_layers_list += [
            self.features[4][idx_1].block[idx_2][2] for idx_1 in range(5) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[4][idx].block[2].activation for idx in range(5)
        ]

        silu_layers_list += [
            self.features[5][idx_1].block[idx_2][2] for idx_1 in range(5) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[5][idx].block[2].activation for idx in range(5)
        ]

        silu_layers_list += [
            self.features[6][idx_1].block[idx_2][2] for idx_1 in range(6) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[6][idx].block[2].activation for idx in range(6)
        ]

        silu_layers_list += [
            self.features[7][idx_1].block[idx_2][2] for idx_1 in range(2) for idx_2 in range(2)
        ]

        silu_layers_list += [
            self.features[7][idx].block[2].activation for idx in range(2)
        ]

        silu_layers_list += [
            self.features[8][2]
        ]

        return silu_layers_list


    
class EfficientNetB4Backbone(torch.nn.Module):
    """
    Standard EfficientNet-B4 feature backbone module.
    Code interface from PyTorch.
    """


    def __init__(
            self,
            batchnorm_momentum=None,
            batchnorm_track_runnning_stats=None,
            batchnorm_eps=None
            ):
                
        super(EfficientNetB4Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        net = torchvision.models.efficientnet_b4(weights=weights)
        
        self.features = net.features

        # BatchNorm mods

        batchnorm_layers_list = self._get_batchnorm_layers()
            
        if batchnorm_momentum is not None:
            for layer in batchnorm_layers_list:
                layer.momentum = batchnorm_momentum
            
        if batchnorm_track_runnning_stats is not None:
            for layer in batchnorm_layers_list:
                layer.track_running_stats = batchnorm_track_runnning_stats
            
        if batchnorm_eps is not None:
            for layer in batchnorm_layers_list:
                layer.eps = batchnorm_eps

        # Feature shape

        self.feature_shape = (1792, 12, 12)


    def forward(self, x):
        
        x = self.features(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.EfficientNet_B4_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform


    def _get_batchnorm_layers(self):

        batchnorm_layers_list = []

        batchnorm_layers_list += [
            self.features[0][1]
        ]
    
        batchnorm_layers_list += [
            self.features[1][idx_1].block[idx_2][1] for idx_1 in range(2) for idx_2 in [0, 2]
        ]
    
        batchnorm_layers_list += [
            self.features[2][idx_1].block[idx_2][1] for idx_1 in range(4) for idx_2 in [0, 1, 3]
        ]
    
        batchnorm_layers_list += [
            self.features[3][idx_1].block[idx_2][1] for idx_1 in range(4) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[4][idx_1].block[idx_2][1] for idx_1 in range(6) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[5][idx_1].block[idx_2][1] for idx_1 in range(6) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[6][idx_1].block[idx_2][1] for idx_1 in range(8) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[7][idx_1].block[idx_2][1] for idx_1 in range(2) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[8][1]
        ]

        return batchnorm_layers_list



class EfficientNetB5Backbone(torch.nn.Module):
    """
    Standard EfficientNet-B5 feature backbone module.
    Code interface from PyTorch.
    """


    def __init__(
            self,
            batchnorm_momentum=None,
            batchnorm_track_runnning_stats=None,
            batchnorm_eps=None
            ):
        
        super(EfficientNetB5Backbone, self).__init__()

        # Model construction

        weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
        net = torchvision.models.efficientnet_b5(weights=weights)
        
        self.features = net.features

        # BatchNorm mods

        batchnorm_layers_list = self._get_batchnorm_layers()
            
        if batchnorm_momentum is not None:
            for layer in batchnorm_layers_list:
                layer.momentum = batchnorm_momentum
            
        if batchnorm_track_runnning_stats is not None:
            for layer in batchnorm_layers_list:
                layer.track_running_stats = batchnorm_track_runnning_stats
            
        if batchnorm_eps is not None:
            for layer in batchnorm_layers_list:
                layer.eps = batchnorm_eps

        # Feature shape

        self.feature_shape = (2048, 15, 15)


    def forward(self, x):
        
        x = self.features(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.EfficientNet_B5_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform


    def _get_batchnorm_layers(self):

        batchnorm_layers_list = []

        batchnorm_layers_list += [
            self.features[0][1]
        ]

        batchnorm_layers_list += [
            self.features[1][idx_1].block[idx_2][1] for idx_1 in range(3) for idx_2 in [0, 2]
        ]

        batchnorm_layers_list += [
            self.features[2][idx_1].block[idx_2][1] for idx_1 in range(5) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[3][idx_1].block[idx_2][1] for idx_1 in range(5) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[4][idx_1].block[idx_2][1] for idx_1 in range(7) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[5][idx_1].block[idx_2][1] for idx_1 in range(7) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[6][idx_1].block[idx_2][1] for idx_1 in range(9) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[7][idx_1].block[idx_2][1] for idx_1 in range(3) for idx_2 in [0, 1, 3]
        ]

        batchnorm_layers_list += [
            self.features[8][1]
        ]

        return batchnorm_layers_list



class EfficientNetV2SmallBackbone(torch.nn.Module):
    """
    Standard EfficientNet V2 Small feature backbone module.
    Code interface from PyTorch.
    """


    def __init__(
            self,
            img_size=None
        ):
        
        super(EfficientNetV2SmallBackbone, self).__init__()

        self.img_size = img_size if img_size is not None else 384

        # Model construction

        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        net = torchvision.models.efficientnet_v2_s(weights=weights)
        
        self.features = net.features

        # Feature shape

        feature_hw = ((self.img_size - 1) // 32) + 1
        self.feature_shape = (1280, feature_hw, feature_hw)

    
    def forward(self, x):
        
        x = self.features(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
        image_transform.antialias = True
        image_transform.crop_size = [self.img_size]
        image_transform.resize_size = [self.img_size]

        return image_transform



class ConvNeXtTinyBackboneOLD(torch.nn.Module):
    """
    Standard ConvNeXt Tiny feature backbone module.
    Code interface from PyTorch.
    """


    def __init__(
            self,
            contiguous_after_permute=False
        ):
        
        super(ConvNeXtTinyBackboneOLD, self).__init__()

        # Model construction

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        net = torchvision.models.convnext_tiny(weights=weights)
        
        self.features = net.features

        # contiguous mods

        if contiguous_after_permute:
            self._add_contiguous_after_permute()
            self._add_contiguous_layernorm2d_end()

        # Feature shape

        self.out_shape = (768, 7, 7)
        self.feature_shape = (768, 7, 7)


    def forward(self, x):
        
        x = self.features(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform


    def _add_contiguous_after_permute(self):

        for idx in range(3):

            self.features[1][idx].block = torch.nn.Sequential(
                self.features[1][idx].block[0],
                self.features[1][idx].block[1],
                Contiguous(),
                self.features[1][idx].block[2],
                self.features[1][idx].block[3],
                self.features[1][idx].block[4],
                self.features[1][idx].block[5],
                self.features[1][idx].block[6],
                Contiguous()
            )

        for idx in range(3):

            self.features[3][idx].block = torch.nn.Sequential(
                self.features[3][idx].block[0],
                self.features[3][idx].block[1],
                Contiguous(),
                self.features[3][idx].block[2],
                self.features[3][idx].block[3],
                self.features[3][idx].block[4],
                self.features[3][idx].block[5],
                self.features[3][idx].block[6],
                Contiguous()
            )

        for idx in range(9):

            self.features[5][idx].block = torch.nn.Sequential(
                self.features[5][idx].block[0],
                self.features[5][idx].block[1],
                Contiguous(),
                self.features[5][idx].block[2],
                self.features[5][idx].block[3],
                self.features[5][idx].block[4],
                self.features[5][idx].block[5],
                self.features[5][idx].block[6],
                Contiguous()
            )

        for idx in range(3):

            self.features[7][idx].block = torch.nn.Sequential(
                self.features[7][idx].block[0],
                self.features[7][idx].block[1],
                Contiguous(),
                self.features[7][idx].block[2],
                self.features[7][idx].block[3],
                self.features[7][idx].block[4],
                self.features[7][idx].block[5],
                self.features[7][idx].block[6],
                Contiguous()
            )


    def _add_contiguous_layernorm2d_end(self):

        new_layernorm2d = LayerNorm2dContiguous(96, eps=1e-6, elementwise_affine=True)
        new_layernorm2d.weight = self.features[0][1].weight
        new_layernorm2d.bias = self.features[0][1].bias
        self.features[0][1] = new_layernorm2d

        new_layernorm2d = LayerNorm2dContiguous(96, eps=1e-6, elementwise_affine=True)
        new_layernorm2d.weight = self.features[2][0].weight
        new_layernorm2d.bias = self.features[2][0].bias
        self.features[2][0] = new_layernorm2d

        new_layernorm2d = LayerNorm2dContiguous(192, eps=1e-6, elementwise_affine=True)
        new_layernorm2d.weight = self.features[4][0].weight
        new_layernorm2d.bias = self.features[4][0].bias
        self.features[4][0] = new_layernorm2d

        new_layernorm2d = LayerNorm2dContiguous(384, eps=1e-6, elementwise_affine=True)
        new_layernorm2d.weight = self.features[6][0].weight
        new_layernorm2d.bias = self.features[6][0].bias
        self.features[6][0] = new_layernorm2d



class ConvNeXtTinyBackbone(torch.nn.Module):
    """
    Standard ConvNeXt Tiny feature backbone module.

    :param img_size: int
        Expected input image size of the backbone.
    """


    def __init__(
            self,
            img_size=None
        ):
        
        super(ConvNeXtTinyBackbone, self).__init__()

        self.img_size = img_size if img_size is not None else 224

        # Model construction

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        net = torchvision.models.convnext_tiny(weights=weights)
        
        self.features = net.features

        # Feature shapes

        feature_hw_7 = min(((self.img_size - 32) // 32) + 8, 7)

        self.feature_shape = (768, feature_hw_7, feature_hw_7)


    def forward(self, x):
        
        x = self.features(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform
