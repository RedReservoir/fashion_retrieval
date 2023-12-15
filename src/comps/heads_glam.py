import torch
import torchvision

import global_local_attention_module_pytorch as glam_pytorch



class RetrievalHeadGLAM(torch.nn.Module):
    """
    Image retrieval embedding module.
    Uses only one convolutional feature level.
    Applies GLAM module to refine the convolutional features.
    
    :param in_feat_shape: tuple
        3-tuple (C, H, W) with the input feature shape.
    :param emb_size: int
        Desired output embedding size.
    :param glam_int_channels: int
        Number of internal channels in GLAM module.
    :param glam_1d_kernel_size: int
        Kernel size of 1D convolution in GLAM module.
    """

    def __init__(
        self,
        in_feat_shape,
        emb_size,
        glam_int_channels,
        glam_1d_kernel_size=3
    ):

        super(RetrievalHeadGLAM, self).__init__()

        self.glam = glam_pytorch.GLAM(
            in_channels=in_feat_shape[0],
            feature_map_size=in_feat_shape[1],
            num_reduced_channels=glam_int_channels,
            kernel_size=glam_1d_kernel_size
        )
        
        self.conv1x1 = torch.nn.Conv2d(
            in_channels=in_feat_shape[0],
            out_channels=emb_size,
            kernel_size=1,
            bias=False
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        
        x = self.glam(x)
        x = self.conv1x1(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x



class RetrievalGLAMHeadPyramidTopDownInstantSimple(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Top-Down, earlier levels are downsampled to match later levels.
    Downsampling is done with average pooling.
    All levels are downsampled to the lstest level at the same time.
    Channel size is reduced with a 1x1 convolution.
    Applies GLAM module to refine the convolutional features.

    :param feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_size: int
        Desired output embedding size.
    :param glam_int_channels_list: list of int
        Number of internal GLAM channels for each used input feature.
        Must have the same length as parameter in_feat_idxs.
    :param glam_1d_kernel_size: int, default=3
        Internal GLAM 1D convolution kernel size.
    :param conv1_groups: int, default=1
        Number of groups to use in the 1x1 Convolutional layer at the end.
        If None is provided, no 1x1 Convolutional layer will be used. This option is only
            available when the embedding size is the same as the input number of channels.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_size,
            glam_int_channels_list,
            glam_1d_kernel_size=3,
            conv1_groups=1
        ):
        
        super(RetrievalGLAMHeadPyramidTopDownInstantSimple, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        # GLAM embellishing & downscaling

        min_feat_h = min(feat_shapes[in_feat_idx][1] for in_feat_idx in in_feat_idxs)
        min_feat_w = min(feat_shapes[in_feat_idx][2] for in_feat_idx in in_feat_idxs)

        self.adapter_layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                glam_pytorch.GLAM(
                    in_channels=feat_shapes[in_feat_idx][0],
                    feature_map_size=feat_shapes[in_feat_idx][1],
                    num_reduced_channels=glam_int_channels_list[in_feat_idx],
                    kernel_size=glam_1d_kernel_size
                ),
                torch.nn.AdaptiveAvgPool2d(
                    output_size=(min_feat_h, min_feat_w)
                )
            )
            for in_feat_idx in in_feat_idxs
        )

        # 1x1 Conv and Avg Pool

        num_total_channels = sum(feat_shapes[in_feat_idx][0] for in_feat_idx in in_feat_idxs)

        if conv1_groups is None:
            self.conv_1x1 = torch.nn.Identity()
        else:
            self.conv_1x1 = torch.nn.Conv2d(num_total_channels, emb_size, kernel_size=1, bias=True, groups=conv1_groups)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
        
        # Instant GLAM + downscale + concat + conv 1x1
        
        new_in_feats = [
            self.adapter_layers[zidx](feats[idx])
            for zidx, idx in enumerate(self._in_feat_idxs)
        ]

        x = torch.cat(new_in_feats, dim=1)
        x = self.conv_1x1(x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x