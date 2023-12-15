import torch
import torchvision



class RetrievalPyramidHead(torch.nn.Module):
    """
    Image retrieval embedding module for Pyramid features.
    Feature levels are upsampled to the highest size before concatenating.

    :param in_feat_idxs: list of int
        List of indices denoting which input features to use.
    :param in_feat_shapes: list of tuple
        List of 3-tuples (C, H, W) with the input feature shapes.
        Must contain as many elements as there are in parameter in_feat_idxs.
    :param emb_size: int
        Desired output embedding size.
    """


    def __init__(
            self,
            in_feat_idxs,
            feat_shapes,
            emb_size
        ):
        
        super(RetrievalPyramidHead, self).__init__()

        self._in_feat_idxs = in_feat_idxs

        # Upsampling

        in_feat_shapes = [feat_shapes[idx] for idx in in_feat_idxs]

        max_feat_h = max(feat_shape[1] for feat_shape in in_feat_shapes)
        max_feat_w = max(feat_shape[2] for feat_shape in in_feat_shapes)

        self.upscale_list = torch.nn.ModuleList(
            torch.nn.Upsample(size=(max_feat_h, max_feat_w), mode="nearest")
            for _ in in_feat_idxs
        )

        # 1x1 Conv and Avg Pool

        num_channels = sum(feat_shape[0] for feat_shape in in_feat_shapes)

        self.conv_1x1 = torch.nn.Conv2d(num_channels, emb_size, kernel_size=1, bias=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, in_feats):
        
        x = torch.cat(
            [
                self.upscale_list[zidx](in_feats[idx])
                for zidx, idx in enumerate(self._in_feat_idxs)
            ],
            dim=1
        )

        x = self.conv_1x1(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)

        return x
    