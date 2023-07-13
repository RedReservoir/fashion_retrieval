import torch
import torchvision

import arch.heads



class RetModel(torch.nn.Module):


    def __init__(self, backbone, emb_size):
        
        super(RetModel, self).__init__()

        self.backbone = backbone
        self.ret_head = heads.RetHead(backbone.out_shape, emb_size)


    def forward(self, x):
        
        x = self.backbone(x)
        x_ret = self.ret_head(x)

        return x_ret
    

    def freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False
    

    def unfreeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = True


    

class ClsRetModel(torch.nn.Module):


    def __init__(self, backbone, num_classes, emb_size):
        
        super(ClsRetModel, self).__init__()

        self.backbone = backbone
        self.cls_head = arch.heads.ClsHead(backbone.out_shape, num_classes)
        self.ret_head = arch.heads.RetHead(backbone.out_shape, emb_size)


    def forward(self, x):
        
        x = self.backbone(x)
        x_cls = self.cls_head(x)
        x_ret = self.ret_head(x)

        return x_cls, x_ret
    

    def freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False
    

    def unfreeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = True
    