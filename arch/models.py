import torch
import torchvision

from arch import heads



class BackboneAndHead(torch.nn.Module):


    def __init__(self, backbone, head):
        
        super(RetModel, self).__init__()

        self.backbone = backbone
        self.head = head


    def forward(self, x):
        
        x = self.backbone(x)
        x = self.head(x)

        return x
    

    def freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False
    

    def unfreeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = True



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


    

class ClsModel(torch.nn.Module):


    def __init__(self, backbone, num_classes):
        
        super(ClsModel, self).__init__()

        self.backbone = backbone
        self.cls_head = heads.ClsHead(backbone.out_shape, num_classes)


    def forward(self, x):
        
        x = self.backbone(x)
        x_cls = self.cls_head(x)

        return x_cls
    

    def freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False
    

    def unfreeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = True
    