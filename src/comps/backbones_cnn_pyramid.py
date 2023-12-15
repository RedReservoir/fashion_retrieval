import torch
import torchvision

import transformers



class ConvNeXtTinyPyramidBackbone(torch.nn.Module):
    """
    Pyramid ConvNeXt Tiny feature backbone module with Pyramid features.

    Yields 4 levels of features:
      - Level 1: Feature blocks 0 and 1.
      - Level 2: Feature blocks 2 and 3.
      - Level 3: Feature blocks 4 and 5.
      - Level 4: Feature blocks 6 and 7.

    :param img_size: int
        Expected input image size of the backbone.
    """


    def __init__(
            self,
            img_size=None
        ):
        
        super(ConvNeXtTinyPyramidBackbone, self).__init__()

        self.img_size = img_size if img_size is not None else 224

        # Model construction

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        net = torchvision.models.convnext_tiny(weights=weights)
        
        self.feature_block_01 = torch.nn.Sequential(net.features[0], net.features[1])
        self.feature_block_23 = torch.nn.Sequential(net.features[2], net.features[3])
        self.feature_block_45 = torch.nn.Sequential(net.features[4], net.features[5])
        self.feature_block_67 = torch.nn.Sequential(net.features[6], net.features[7])

        # Feature shapes

        feature_hw_1 = min(self.img_size // 4, 56)
        feature_hw_3 = min(self.img_size // 8, 28)
        feature_hw_5 = min(self.img_size // 16, 14)
        feature_hw_7 = min(self.img_size // 32, 7)

        self.feature_shapes = [
            (96, feature_hw_1, feature_hw_1),
            (192, feature_hw_3, feature_hw_3),
            (384, feature_hw_5, feature_hw_5),
            (768, feature_hw_7, feature_hw_7)
        ]


    def forward(self, x):
        
        x_1 = self.feature_block_01(x)
        x_3 = self.feature_block_23(x_1)
        x_5 = self.feature_block_45(x_3)
        x_7 = self.feature_block_67(x_5)

        return x_1, x_3, x_5, x_7
    

    def get_image_transform(self):

        image_transform = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()
        image_transform.antialias = True

        return image_transform



class ConvNeXtV2TinyMultilevelBackbone(torch.nn.Module):
    """
    Multilevel ConvNeXt V2 Tiny feature backbone module with multilevel features.

    Yields 4 levels of features:
      - Level 1: Feature stage 1.
      - Level 2: Feature stage 2.
      - Level 3: Feature stage 3.
      - Level 4: Feature stage 4.

    :param img_size: int
        Expected input image size of the backbone.
    """



    class ConvNeXtV2TinyMultilevelImageTransform:
        """
        Image transformation object for ConvNeXt V2 Tiny feature backbone.
        """


        def __init__(
                self,
                img_size
            ):
            
            self.image_transform = transformers.AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-22k-224")
            self.image_transform.size["shortest_edge"] = img_size


        def __call__(self, img_tensor):
            
            return self.image_transform(img_tensor, return_tensors="pt").pixel_values[0]
        


    def __init__(
            self,
            img_size=None
        ):
        
        super(ConvNeXtV2TinyMultilevelBackbone, self).__init__()

        self.img_size = img_size if img_size is not None else 224

        # Model construction

        net = transformers.ConvNextV2Model.from_pretrained("facebook/convnextv2-tiny-22k-224")
        
        self.embeddings = net.embeddings
        self.encoder_stage_1 = net.encoder.stages[0]
        self.encoder_stage_2 = net.encoder.stages[1]
        self.encoder_stage_3 = net.encoder.stages[2]
        self.encoder_stage_4 = net.encoder.stages[3]

        # Feature shapes

        feature_hw_1 = self.img_size // 4
        feature_hw_2 = self.img_size // 8
        feature_hw_3 = self.img_size // 16
        feature_hw_4 = self.img_size // 32

        self.feature_shapes = [
            (96, feature_hw_1, feature_hw_1),
            (192, feature_hw_2, feature_hw_2),
            (384, feature_hw_3, feature_hw_3),
            (768, feature_hw_4, feature_hw_4)
        ]


    def forward(self, x):
        
        x = self.embeddings(x)
        x_1 = self.encoder_stage_1(x)
        x_2 = self.encoder_stage_2(x_1)
        x_3 = self.encoder_stage_3(x_2)
        x_4 = self.encoder_stage_4(x_3)

        return x_1, x_2, x_3, x_4
    

    def get_image_transform(self):

        return self.ConvNeXtV2TinyMultilevelImageTransform(self.img_size)
