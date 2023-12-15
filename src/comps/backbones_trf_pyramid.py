import os
import pathlib

from typing import List

import torch
import torchvision

import transformers
import src.external.gc_vit.gc_vit as ext_gc_vit



class GCVitTinyMultilevelBackbone(torch.nn.Module):
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



    class GCViTLayerNoDown(torch.nn.Module):
        """
        """


        def __init__(self, q_global_gen, blocks):
            super(GCVitTinyMultilevelBackbone.GCViTLayerNoDown, self).__init__()
            self.q_global_gen = q_global_gen
            self.blocks = blocks


        def forward(self, x):
            q_global = self.q_global_gen(x.permute(0, 3, 1, 2).contiguous())
            for blk in self.blocks: x = blk(x, q_global)
            return x



    class GCVitTinyMultilevelImageTransform:
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
            with_layernorm=False,
            img_size=None,
            ckp_filename=os.path.join(pathlib.Path.home(), "data", "pretrained_weights", "gcvit_1k_tiny.pth.tar")
        ):
        
        super(GCVitTinyMultilevelBackbone, self).__init__()

        # Argument checking

        self.img_size = img_size if img_size is not None else 224

        # Model construction

        net = ext_gc_vit.gc_vit_tiny(pretrained=False)

        ckp = torch.load(ckp_filename, map_location="cpu")
        net.load_state_dict(ckp["state_dict"])

        self.input_layers = torch.nn.Sequential(
            net.patch_embed,
            net.pos_drop
        )

        self.level_1_layer = self.GCViTLayerNoDown(
            net.levels[0].q_global_gen,
            net.levels[0].blocks
        )

        self.level_2_layer = self.GCViTLayerNoDown(
            net.levels[1].q_global_gen,
            net.levels[1].blocks
        )

        self.level_3_layer = self.GCViTLayerNoDown(
            net.levels[2].q_global_gen,
            net.levels[2].blocks
        )

        self.level_4_layer = self.GCViTLayerNoDown(
            net.levels[3].q_global_gen,
            net.levels[3].blocks
        )

        self.level_1_down = net.levels[0].downsample

        self.level_2_down = net.levels[1].downsample

        self.level_3_down = net.levels[2].downsample

        if with_layernorm:

            self.level_1_ln = torch.nn.LayerNorm((64,), eps=1e-5, elementwise_affine=True)
            self.level_2_ln = torch.nn.LayerNorm((128,), eps=1e-5, elementwise_affine=True)
            self.level_3_ln = torch.nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True)
            self.level_4_ln = net.norm

        else:

            self.level_1_ln = torch.nn.Identity()
            self.level_2_ln = torch.nn.Identity()
            self.level_3_ln = torch.nn.Identity()
            self.level_4_ln = torch.nn.Identity()
        
        self.level_1_permute = self.Permute(0, 3, 1, 2)
        self.level_2_permute = self.Permute(0, 3, 1, 2)
        self.level_3_permute = self.Permute(0, 3, 1, 2)
        self.level_4_permute = self.Permute(0, 3, 1, 2)

        # Feature shape

        self.feature_shapes = [
            (64, 56, 56),
            (128, 28, 28),
            (256, 14, 14),
            (512, 7, 7)
        ]


    def forward(self, x):
        
        x = self.input_layers(x)
        x_1 = self.level_1_layer(x)
        x = self.level_1_down(x_1)
        x_2 = self.level_2_layer(x)
        x = self.level_2_down(x_2)
        x_3 = self.level_3_layer(x)
        x = self.level_3_down(x_3)
        x_4 = self.level_4_layer(x)

        x_1 = self.level_1_ln(x_1)
        x_2 = self.level_2_ln(x_2)
        x_3 = self.level_3_ln(x_3)
        x_4 = self.level_4_ln(x_4)

        x_1 = self.level_1_permute(x_1)
        x_2 = self.level_2_permute(x_2)
        x_3 = self.level_3_permute(x_3)
        x_4 = self.level_4_permute(x_4)

        return x_1, x_2, x_3, x_4
    

    def get_image_transform(self):

        return self.GCVitTinyMultilevelImageTransform(self.img_size)
