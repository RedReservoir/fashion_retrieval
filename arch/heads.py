import torch
import torchvision



class RetHead(torch.nn.Module):
    """
    Generic image retrieval embedding module.

    :param in_feat_shape: tuple
        3-tuple (C, H, W) with the input feature shape.
    :param emb_size: int
        Desired output embedding size.
    """

    def __init__(self, in_feat_shape, emb_size):
        
        super(RetHead, self).__init__()

        self.ret_conv1x1 = self.conv1x1(in_feat_shape[0], emb_size)
        self.cls_avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        
        x = self.ret_conv1x1(x)
        x = self.cls_avgpool(x)
        x = torch.squeeze(x)

        return x
    

    def conv1x1(self, in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:

        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    


class ClsHead(torch.nn.Module):
    """
    Generic image classification module.

    :param in_feat_shape: tuple
        3-tuple (C, H, W) with the input feature shape.
    :param num_classes: int
        Number of classes.
        Essentially the desired output embedding size.
    """


    def __init__(self, in_feat_shape, num_classes):
        
        super(ClsHead, self).__init__()

        self.cls_avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.cls_linear = torch.nn.Linear(in_feat_shape[0], num_classes)
        self.cls_smax = torch.nn.Softmax()


    def forward(self, x):
        
        x = self.cls_avgpool(x)
        x = torch.squeeze(x)
        x = self.cls_linear(x)
        x = self.cls_smax(x)

        return x
    