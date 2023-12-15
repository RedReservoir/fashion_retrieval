import os
import pathlib

from typing import List

import torch
import torchvision

import transformers
import fastervit
import src.external.gc_vit.gc_vit as ext_gc_vit



class SwinTransformerV2TinyBackbone(torch.nn.Module):
    """
    Standard SwinT V2 Tiny feature backbone module.
    Code interface from PyTorch.
    """



    class LayerContiguous(torch.nn.Module):
        """
        Wrapper module that applies torch.contiguous() to a tensor after another module.
        """


        def __init__(self, prev_module):
            super().__init__()
            self.prev_module = prev_module

        def forward(self, *args, **kwargs) -> torch.Tensor:
            return self.prev_module(*args, **kwargs).contiguous()



    def __init__(
            self,
            img_size=None,
            contiguous_after_permute=False
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

        # Contiguous mods

        if contiguous_after_permute:
            self._add_contiguous_after_permute()

    
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


    def _add_contiguous_after_permute(self):

        self.features[0][1] = self.LayerContiguous(self.features[0][1])
        self.permute = self.LayerContiguous(self.permute)



class CvT21Backbone(torch.nn.Module):
    """
    Standard CvT21 feature backbone module.
    Code interface from HuggingFace.
    """
        


    class CvT21ImageTransform:
        """
        Image transformation object for CvT21 feature backbone.
        """


        def __init__(
                self,
                pretrn_name,
                img_size
            ):
            
            self.image_transform = transformers.AutoImageProcessor.from_pretrained(pretrn_name)
            self.image_transform.size["shortest_edge"] = img_size


        def __call__(self, img_tensor):
            
            return self.image_transform(img_tensor, return_tensors="pt").pixel_values[0]



    def __init__(
            self,
            pretrn_name=None,
            img_size=None
        ):
        
        super(CvT21Backbone, self).__init__()

        # Argument checker

        self.pretrn_name = pretrn_name if pretrn_name is not None else "microsoft/cvt-21"
        
        pretrn_name_to_img_size_dict = {
            "microsoft/cvt-21": 224,
            "microsoft/cvt-21-384": 384,
            "microsoft/cvt-21-384-22k": 384
        }
        
        self.img_size = img_size if img_size is not None else pretrn_name_to_img_size_dict[self.pretrn_name]

        # Model construction

        net = transformers.CvtModel.from_pretrained(self.pretrn_name)
        self.net = net

        # Feature shape

        feature_hw = (self.img_size + 13) // 16
        self.feature_shape = (384, feature_hw, feature_hw)

    
    def forward(self, x):
        
        x = self.net(x)

        return x.last_hidden_state
    

    def get_image_transform(self):

        return self.CvT21ImageTransform(
            self.pretrn_name,
            self.img_size
        )



class GCVitTinyBackbone(torch.nn.Module):
    """
    Standard GCVit Tiny feature backbone module.
    Pre-trained weights obtained from https://github.com/NVlabs/GCViT.

    Image size must be fixed at 224.
    """



    class Permute(torch.nn.Module):
        """
        Wrapper module that applies torch.contiguous() to a tensor after another module.
        """


        def __init__(self, *permute_dims: List[int]):
            super().__init__()
            self.permute_dims = list(permute_dims)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.permute(x, self.permute_dims)



    class GCVitTinyImageTransform:
        """
        Image transformation object for GCVit Tiny feature backbone.
        """


        def __init__(
            self,
            img_size
        ):
            
            self.resize_fn = torchvision.transforms.Resize((img_size, img_size), antialias=True)
            self.normalize_fn = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        def __call__(self, img_tensor):
            
            img_tensor = self.resize_fn(img_tensor)
            img_tensor = img_tensor.float() / 255
            img_tensor = self.normalize_fn(img_tensor)
            
            return img_tensor



    def __init__(
            self,
            img_size=None,
            ckp_filename=os.path.join(pathlib.Path.home(), "data", "pretrained_weights", "gcvit_1k_tiny.pth.tar")
        ):
        
        super(GCVitTinyBackbone, self).__init__()

        # Argument checking

        self.img_size = img_size if img_size is not None else 224

        # Model construction

        net = ext_gc_vit.gc_vit_tiny(pretrained=False)

        ckp = torch.load(ckp_filename, map_location="cpu")
        net.load_state_dict(ckp["state_dict"])

        self.backbone = torch.nn.Sequential(
            net.patch_embed,
            net.pos_drop,
            net.levels[0],
            net.levels[1],
            net.levels[2],
            net.levels[3],
            net.norm
        )

        self.permute = self.Permute(0, 3, 1, 2)

        # Feature shape

        self.feature_shape = (512, 7, 7)


    def forward(self, x):
        
        x = self.backbone(x)
        x = self.permute(x)

        return x
    

    def get_image_transform(self):

        return self.GCVitTinyImageTransform(self.img_size)



class FasterVit0Backbone(torch.nn.Module):
    """
    Standard Faster-Vit 0 feature backbone module.

    Image size must be fixed at 224.
    """



    class FasterVit0ImageTransform:
        """
        Image transformation object for Faster-Vit 0 feature backbone.
        """


        def __init__(
            self,
            img_size
        ):
            
            self.resize_fn = torchvision.transforms.Resize((img_size, img_size), antialias=True)
            self.normalize_fn = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        def __call__(self, img_tensor):
            
            img_tensor = self.resize_fn(img_tensor)
            img_tensor = img_tensor.float() / 255
            img_tensor = self.normalize_fn(img_tensor)
            
            return img_tensor



    def __init__(
            self,
            img_size=None
        ):
        
        super(FasterVit0Backbone, self).__init__()

        # Argument checking

        self.img_size = img_size if img_size is not None else 224

        # Model construction

        net = fastervit.create_model(
            "faster_vit_0_224",
            pretrained=True
        )

        self.backbone = torch.nn.Sequential(
            net.patch_embed,
            net.levels[0],
            net.levels[1],
            net.levels[2],
            net.levels[3],
            net.norm
        )

        # Feature shape

        self.feature_shape = (512, 7, 7)


    def forward(self, x):
        
        x = self.backbone(x)

        return x
    

    def get_image_transform(self):

        return self.FasterVit0ImageTransform(self.img_size)

