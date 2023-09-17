import torch
import torchvision

import transformers



class SwinTransformerV2TinyBackbone(torch.nn.Module):
    """
    Standard SwinT V2 Tiny feature backbone module.
    Code interface from PyTorch.
    """


    def __init__(
            self,
            img_size=None
        ):
        
        super(SwinTransformerV2TinyBackbone, self).__init__()

        self.img_size = img_size if img_size is not None else 256

        # Model construction

        weights = torchvision.models.Swin_V2_T_Weights.DEFAULT
        net = torchvision.models.swin_v2_t(weights=weights)
        
        self.features = net.features
        self.norm = net.norm
        self.permute = net.permute

        # Feature shape

        feature_hw = ((self.img_size - 4) // 32) + 1
        self.feature_shape = (768, feature_hw, feature_hw)

    
    def forward(self, x):
        
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)

        return x
    

    def get_image_transform(self):

        image_transform = torchvision.models.Swin_V2_T_Weights.DEFAULT.transforms()
        image_transform.antialias = True
        image_transform.crop_size = [self.img_size]
        image_transform.resize_size = [round(self.img_size * 260 / 256)]

        return image_transform



class CvTransformerB21I384D22kBackbone(torch.nn.Module):
    """
    Standard CvT-21-384-22k feature backbone module.
    Code interface from HuggingFace.
    """


    def __init__(
            self,
            img_size=None
        ):
        
        super(CvTransformerB21I384D22kBackbone, self).__init__()

        self.img_size = img_size if img_size is not None else 384

        # Model construction

        net = transformers.CvtModel.from_pretrained("microsoft/cvt-21-384-22k")
        
        self.net = net

        # Feature shape

        feature_hw = ((self.img_size - 3) // 16) + 1
        self.feature_shape = (384, feature_hw, feature_hw)

    
    def forward(self, x):
        
        x = self.net(x)

        return x.last_hidden_state
    

    def get_image_transform(self):

        return self._image_transform

        #CONTINUE HERE

        image_transform = transformers.AutoImageProcessor.from_pretrained("microsoft/cvt-21-384-22k")
        image_transform.size["shortest_edge"] = self.img_size
        image_transform_corr = lambda t: torch.from_numpy(image_transform(t).pixel_values[0])

        return image_transform_corr


    def _image_transform(t):

        image_transform = transformers.AutoImageProcessor.from_pretrained("microsoft/cvt-21-384-22k")
        image_transform.size["shortest_edge"] = self.img_size
        t = torch.from_numpy(image_transform(t).pixel_values[0])
        return t