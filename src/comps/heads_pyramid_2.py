import math

import torch
import torchvision

import src.utils.math
import src.utils.tensor



class RetrievalHeadPyramidBottomUpInstantSimple(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Bottom-Up, later levels are upsampled to match earlier levels.
    Upsampling is done with bilinear interpolation.
    All levels are upsampled to the earliest level at the same time.
    Channel size is reduced with a 1x1 convolution.

    :param feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_size: int
        Desired output embedding size.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_size
        ):
        
        super(RetrievalHeadPyramidBottomUpInstantSimple, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]

        # Upsampling

        max_feat_h = max(feat_shape[1] for feat_shape in in_feat_shapes)
        max_feat_w = max(feat_shape[2] for feat_shape in in_feat_shapes)

        self.upscale_layers = torch.nn.ModuleList(
            torch.nn.Upsample(size=(max_feat_h, max_feat_w), mode="bilinear")
            for _ in in_feat_idxs
        )

        # 1x1 Conv and Avg Pool

        num_total_channels = sum(feat_shape[0] for feat_shape in in_feat_shapes)

        self.conv_1x1 = torch.nn.Conv2d(num_total_channels, emb_size, kernel_size=1, bias=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
        
        # Instant upscale + concat + conv 1x1
        
        new_in_feats = [
            self.upscale_layers[zidx](feats[idx])
            for zidx, idx in enumerate(self._in_feat_idxs)
        ]

        x = torch.cat(new_in_feats, dim=1)
        x = self.conv_1x1(x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
    


class RetrievalHeadPyramidBottomUpProgressiveSimple(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Bottom-Up, later levels are upsampled to match earlier levels.
    Upsampling is done with bilinear interpolation.
    Levels are progressively upsampled to the previous level.
    Channel size is modified with a 1x1 convolution each time.

    :param in_feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_sizes: list of int
        Desired intermediate and output embedding sizes.
        Length of this parameter is expected to be `len(in_feat_idxs) - 1`.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_sizes
        ):
        
        super(RetrievalHeadPyramidBottomUpProgressiveSimple, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]

        # Upsampling

        up_feat_h_list = [feat_shape[1] for feat_shape in in_feat_shapes[-2::-1]]
        up_feat_w_list = [feat_shape[2] for feat_shape in in_feat_shapes[-2::-1]]

        self.upscale_layers = torch.nn.ModuleList(
            torch.nn.Upsample(size=(up_feat_h, up_feat_w), mode="bilinear")
            for up_feat_h, up_feat_w in zip(up_feat_h_list, up_feat_w_list)
        )

        # 1x1 Convolution

        cat_1_channels_list = [feat_shape[0] for feat_shape in in_feat_shapes[-2::-1]]
        cat_2_channels_list = [in_feat_shapes[-1][0]] + emb_sizes[:-1]
        in_channels_list = [
            cat_1_channels + cat_2_channels
            for (cat_1_channels, cat_2_channels)
            in zip(cat_1_channels_list, cat_2_channels_list)
        ]
        
        out_channels_list = emb_sizes

        self.conv_1x1_layers = torch.nn.ModuleList(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels, out_channels in zip(in_channels_list, out_channels_list)
        )
        
        # Final Avg Pooling
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
                
        # Iterative upscale + concat + conv 1x1

        x = feats[self._in_feat_idxs[-1]]

        for zidx, idx in enumerate(self._in_feat_idxs[-2::-1]):

            x = self.upscale_layers[zidx](x)
            x = torch.cat([x, feats[idx]], dim=1)
            x = self.conv_1x1_layers[zidx](x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x



class RetrievalHeadPyramidBottomUpInstantConv(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Implementation specific for ConvNeXt-T
      - Assumes feature maps are spatially square.
      - Assumes feature map spatial size differs in powers of 2.

    Bottom-Up, later levels are upsampled to match earlier levels.
    Upsampling is done with transposed convolutions.
    All levels are upsampled to the earliest level at the same time.
    Channel size is reduced with a 1x1 convolution.

    :param feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_size: int
        Desired output embedding size.
    :param conv_par_perc: float
        Parameter usage percentage for TransposeConv2D layers.
        Must be a value between 0 and 1.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_size,
            conv_par_perc
        ):
        
        super(RetrievalHeadPyramidBottomUpInstantConv, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]
        desired_groups = 1 / conv_par_perc if conv_par_perc != 0 else float("inf")

        # Upsampling

        max_feat_s = max(feat_shape[1] for feat_shape in in_feat_shapes)

        self.upscale_layers = torch.nn.ModuleList()
        for idx in in_feat_idxs:

            num_conv_layers = round(math.log2(max_feat_s / feat_shapes[idx][1]))
            
            if num_conv_layers == 0:
                
                self.upscale_layers.append(torch.nn.Identity())

            else:

                in_channels = feat_shapes[idx][0]
                out_channels = feat_shapes[idx][0]

                groups = src.utils.math.get_closest_perc_div(in_channels, desired_groups)

                self.upscale_layers.append(torch.nn.Sequential(*[
                    torch.nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        output_padding=1,
                        groups=groups
                    )
                    for _ in range(num_conv_layers)
                ]))

        # 1x1 Conv and Avg Pool

        num_total_channels = sum(feat_shape[0] for feat_shape in in_feat_shapes)

        self.conv_1x1 = torch.nn.Conv2d(num_total_channels, emb_size, kernel_size=1, bias=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
        
        # Instant upscale + concat + conv 1x1
        
        new_in_feats = [
            self.upscale_layers[zidx](feats[idx])
            for zidx, idx in enumerate(self._in_feat_idxs)
        ]

        x = torch.cat(new_in_feats, dim=1)
        x = self.conv_1x1(x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
    


class RetrievalHeadPyramidBottomUpProgressiveConv(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.
    
    Implementation specific for ConvNeXt-T
      - Assumes feature maps are spatially square.
      - Assumes feature map spatial size differs in powers of 2.

    Bottom-Up, later levels are upsampled to match earlier levels.
    Upsampling is done with transposed convolutions.
    Levels are progressively upsampled to the previous level.
    Channel size is modified with a 1x1 convolution each time.
    
    :param in_feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_sizes: list of int
        Desired intermediate and output embedding sizes.
        Length of this parameter is expected to be `len(in_feat_idxs) - 1`.
    :param conv_par_perc: float
        Parameter usage percentage for TransposeConv2D layers.
        Must be a value between 0 and 1.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_sizes,
            conv_par_perc
        ):
        
        super(RetrievalHeadPyramidBottomUpProgressiveConv, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]
        desired_groups = 1 / conv_par_perc if conv_par_perc != 0 else float("inf")

        # Upsampling

        num_conv_layers_list = [
            round(math.log2(feat_shapes[in_feat_idxs[zidx]][1] / feat_shapes[in_feat_idxs[zidx+1]][1]))
            for zidx in reversed(range(len(in_feat_idxs) - 1))
        ]

        num_channels_list = [in_feat_shapes[-1][0]] + emb_sizes[:-1]

        self.upscale_layers = torch.nn.ModuleList()
        for zidx in range(len(in_feat_idxs) - 1):

            num_conv_layers = num_conv_layers_list[zidx]
            
            in_channels = num_channels_list[zidx]
            out_channels = num_channels_list[zidx]

            groups = src.utils.math.get_closest_perc_div(in_channels, desired_groups)

            self.upscale_layers.append(torch.nn.Sequential(*[
                torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    output_padding=1,
                    groups=groups
                )
                for _ in range(num_conv_layers)
            ]))

        # 1x1 Convolution

        cat_1_channels_list = [feat_shape[0] for feat_shape in in_feat_shapes[-2::-1]]
        cat_2_channels_list = [in_feat_shapes[-1][0]] + emb_sizes[:-1]
        in_channels_list = [
            cat_1_channels + cat_2_channels
            for (cat_1_channels, cat_2_channels)
            in zip(cat_1_channels_list, cat_2_channels_list)
        ]
        
        out_channels_list = emb_sizes

        self.conv_1x1_layers = torch.nn.ModuleList(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels, out_channels in zip(in_channels_list, out_channels_list)
        )
        
        # Final Avg Pooling
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
                
        # Iterative upscale + concat + conv 1x1

        x = feats[self._in_feat_idxs[-1]]

        for zidx, idx in enumerate(self._in_feat_idxs[-2::-1]):

            x = self.upscale_layers[zidx](x)
            x = torch.cat([x, feats[idx]], dim=1)
            x = self.conv_1x1_layers[zidx](x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x



#



class RetrievalHeadPyramidTopDownInstantSimple(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Top-Down, earlier levels are downsampled to match later levels.
    Downsampling is done with average pooling.
    All levels are downsampled to the lstest level at the same time.
    Channel size is reduced with a 1x1 convolution.

    :param feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_size: int
        Desired output embedding size.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_size
        ):
        
        super(RetrievalHeadPyramidTopDownInstantSimple, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]

        # Downsampling

        min_feat_h = min(feat_shape[1] for feat_shape in in_feat_shapes)
        min_feat_w = min(feat_shape[2] for feat_shape in in_feat_shapes)

        self.downscale_layers = torch.nn.ModuleList(
            torch.nn.AdaptiveAvgPool2d(output_size=(min_feat_h, min_feat_w))
            for _ in in_feat_idxs
        )

        # 1x1 Conv and Avg Pool

        num_total_channels = sum(feat_shape[0] for feat_shape in in_feat_shapes)

        self.conv_1x1 = torch.nn.Conv2d(num_total_channels, emb_size, kernel_size=1, bias=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
        
        # Instant downscale + concat + conv 1x1
        
        new_in_feats = [
            self.downscale_layers[zidx](feats[idx])
            for zidx, idx in enumerate(self._in_feat_idxs)
        ]

        x = torch.cat(new_in_feats, dim=1)
        x = self.conv_1x1(x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
    


class RetrievalHeadPyramidTopDownInstantSimple2(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Top-Down, earlier levels are downsampled to match later levels.
    Downsampling is done with average pooling.
    All levels are downsampled to the lstest level at the same time.
    Channel size is reduced with a 1x1 convolution.

    :param feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_size: int
        Desired output embedding size.
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
            conv1_groups=1
        ):
        
        super(RetrievalHeadPyramidTopDownInstantSimple2, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]

        # Downsampling

        min_feat_h = min(feat_shape[1] for feat_shape in in_feat_shapes)
        min_feat_w = min(feat_shape[2] for feat_shape in in_feat_shapes)

        self.downscale_layers = torch.nn.ModuleList(
            torch.nn.AdaptiveAvgPool2d(output_size=(min_feat_h, min_feat_w))
            for _ in in_feat_idxs
        )

        # 1x1 Conv and Avg Pool

        num_total_channels = sum(feat_shape[0] for feat_shape in in_feat_shapes)

        if conv1_groups is None:
            self.conv_1x1 = torch.nn.Identity()
        else:
            self.conv_1x1 = torch.nn.Conv2d(num_total_channels, emb_size, kernel_size=1, bias=True, groups=conv1_groups)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
        
        # Instant downscale + concat + conv 1x1
        
        new_in_feats = [
            self.downscale_layers[zidx](feats[idx])
            for zidx, idx in enumerate(self._in_feat_idxs)
        ]

        x = torch.cat(new_in_feats, dim=1)
        x = self.conv_1x1(x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
    


class RetrievalHeadPyramidTopDownProgressiveSimple(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Top-Down, earlier levels are downsampled to match later levels.
    Downsampling is done with average pooling.
    Levels are progressively downsampled to the next level.
    Channel size is modified with a 1x1 convolution each time.

    :param in_feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_sizes: list of int
        Desired intermediate and output embedding sizes.
        Length of this parameter is expected to be `len(in_feat_idxs) - 1`.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_sizes
        ):
        
        super(RetrievalHeadPyramidTopDownProgressiveSimple, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]

        # Downsampling

        down_feat_h_list = [feat_shape[1] for feat_shape in in_feat_shapes[1:]]
        down_feat_w_list = [feat_shape[2] for feat_shape in in_feat_shapes[1:]]

        self.downscale_layers = torch.nn.ModuleList(
            torch.nn.AdaptiveAvgPool2d(output_size=(down_feat_h, down_feat_w))
            for down_feat_h, down_feat_w in zip(down_feat_h_list, down_feat_w_list)
        )

        # 1x1 Convolution

        cat_1_channels_list = [in_feat_shapes[0][0]] + emb_sizes[:-1]
        cat_2_channels_list = [feat_shape[0] for feat_shape in in_feat_shapes[1:]]
        in_channels_list = [
            cat_1_channels + cat_2_channels
            for (cat_1_channels, cat_2_channels)
            in zip(cat_1_channels_list, cat_2_channels_list)
        ]
        
        out_channels_list = emb_sizes

        self.conv_1x1_layers = torch.nn.ModuleList(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels, out_channels in zip(in_channels_list, out_channels_list)
        )
        
        # Final Avg Pooling
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
                
        # Iterative downscale + concat + conv 1x1

        x = feats[self._in_feat_idxs[0]]

        for zidx, idx in enumerate(self._in_feat_idxs[1:]):

            x = self.downscale_layers[zidx](x)
            x = torch.cat([x, feats[idx]], dim=1)
            x = self.conv_1x1_layers[zidx](x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
    


class RetrievalHeadPyramidTopDownInstantConv(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.

    Implementation specific for ConvNeXt-T
      - Assumes feature maps are spatially square.
      - Assumes feature map spatial size differs in powers of 2.

    Top-Down, earlier levels are downsampled to match later levels.
    Downsampling is done with convolutions.
    All levels are downsampled to the latest level at the same time.
    Channel size is reduced with a 1x1 convolution.

    :param feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_size: int
        Desired output embedding size.
    :param conv_par_perc: float
        Parameter usage percentage for Conv2D layers.
        Must be a value between 0 and 1.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_size,
            conv_par_perc
        ):
        
        super(RetrievalHeadPyramidTopDownInstantConv, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]
        desired_groups = 1 / conv_par_perc if conv_par_perc != 0 else float("inf")

        # Downsampling

        min_feat_s = min(feat_shape[1] for feat_shape in in_feat_shapes)

        self.downscale_layers = torch.nn.ModuleList()
        for idx in in_feat_idxs:

            num_conv_layers = round(math.log2(feat_shapes[idx][1] / min_feat_s))
            
            if num_conv_layers == 0:
                
                self.downscale_layers.append(torch.nn.Identity())

            else:

                in_channels = feat_shapes[idx][0]
                out_channels = feat_shapes[idx][0]

                groups = src.utils.math.get_closest_perc_div(in_channels, desired_groups)

                self.downscale_layers.append(torch.nn.Sequential(*[
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        groups=groups
                    )
                    for _ in range(num_conv_layers)
                ]))

        # 1x1 Conv and Avg Pool

        num_total_channels = sum(feat_shape[0] for feat_shape in in_feat_shapes)

        self.conv_1x1 = torch.nn.Conv2d(num_total_channels, emb_size, kernel_size=1, bias=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
        
        # Instant downscale + concat + conv 1x1
        
        new_in_feats = [
            self.downscale_layers[zidx](feats[idx])
            for zidx, idx in enumerate(self._in_feat_idxs)
        ]

        x = torch.cat(new_in_feats, dim=1)
        x = self.conv_1x1(x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
    


class RetrievalHeadPyramidTopDownProgressiveConv(torch.nn.Module):
    """
    Image retrieval embedding module for pyramid features.
    
    Implementation specific for ConvNeXt-T
      - Assumes feature maps are spatially square.
      - Assumes feature map spatial size differs in powers of 2.

    Tod-Down, earlier levels are downsampled to match later levels.
    Downsampling is done with convolutions.
    Levels are progressively downsampled to the next level.
    Channel size is modified with a 1x1 convolution each time.
    
    :param in_feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param emb_sizes: list of int
        Desired intermediate and output embedding sizes.
        Length of this parameter is expected to be `len(in_feat_idxs) - 1`.
    :param conv_par_perc: float
        Parameter usage percentage for TransposeConv2D layers.
        Must be a value between 0 and 1.
    """


    def __init__(
            self,
            feat_shapes,
            in_feat_idxs,
            emb_sizes,
            conv_par_perc
        ):
        
        super(RetrievalHeadPyramidTopDownProgressiveConv, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]
        desired_groups = 1 / conv_par_perc if conv_par_perc != 0 else float("inf")

        # Downsampling

        num_conv_layers_list = [
            round(math.log2(feat_shapes[in_feat_idxs[zidx]][1] / feat_shapes[in_feat_idxs[zidx+1]][1]))
            for zidx in range(len(in_feat_idxs) - 1)
        ]

        num_channels_list = [in_feat_shapes[0][0]] + emb_sizes[:-1]

        self.downscale_layers = torch.nn.ModuleList()
        for zidx in range(len(in_feat_idxs) - 1):

            num_conv_layers = num_conv_layers_list[zidx]
            
            in_channels = num_channels_list[zidx]
            out_channels = num_channels_list[zidx]

            groups = src.utils.math.get_closest_perc_div(in_channels, desired_groups)

            self.downscale_layers.append(torch.nn.Sequential(*[
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    groups=groups
                )
                for _ in range(num_conv_layers)
            ]))

        # 1x1 Convolution

        cat_1_channels_list = [in_feat_shapes[0][0]] + emb_sizes[:-1]
        cat_2_channels_list = [feat_shape[0] for feat_shape in in_feat_shapes[1:]]
        in_channels_list = [
            cat_1_channels + cat_2_channels
            for (cat_1_channels, cat_2_channels)
            in zip(cat_1_channels_list, cat_2_channels_list)
        ]
        
        out_channels_list = emb_sizes

        self.conv_1x1_layers = torch.nn.ModuleList(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels, out_channels in zip(in_channels_list, out_channels_list)
        )
        
        # Final Avg Pooling
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, feats):
                
        # Iterative downscale + concat + conv 1x1

        x = feats[self._in_feat_idxs[0]]

        for zidx, idx in enumerate(self._in_feat_idxs[1:]):

            x = self.downscale_layers[zidx](x)
            x = torch.cat([x, feats[idx]], dim=1)
            x = self.conv_1x1_layers[zidx](x)

        # Final Avg Pooling

        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
