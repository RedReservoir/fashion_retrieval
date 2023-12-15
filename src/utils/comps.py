import torch

import src.comps.backbones_cnn
import src.comps.backbones_cnn_pyramid
import src.comps.backbones_trf
import src.comps.heads
import src.comps.heads_pyramid
import src.comps.heads_pyramid_2
import src.comps.heads_glam

import src.utils.train



def create_backbone(backbone_options):
    """
    Creates a backbone for a MTL model.

    :param backbone_options: dict
        Options to create the backbone with.

    :return: torch.nn.module
        The created backbone.
    """

    backbone_class = backbone_options["class"]
    backbone_img_size = backbone_options.get("img_size", None)

    # ResNet

    if backbone_class == "ResNet50Backbone":
        backbone = src.comps.backbones_cnn.ResNet50Backbone()

    # EfficientNet
    
    if backbone_class == "EfficientNetB2Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = src.comps.backbones_cnn.EfficientNetB2Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    if backbone_class == "EfficientNetB3Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = src.comps.backbones_cnn.EfficientNetB3Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    if backbone_class == "EfficientNetB4Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = src.comps.backbones_cnn.EfficientNetB4Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    if backbone_class == "EfficientNetB5Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = src.comps.backbones_cnn.EfficientNetB5Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    # EfficientNet V2

    if backbone_class == "EfficientNetV2SmallBackbone":
        backbone = src.comps.backbones_cnn.EfficientNetV2SmallBackbone(
            backbone_img_size
        )
    
    # ConvNeXt

    if backbone_class == "ConvNeXtTinyBackboneOLD":
        backbone = src.comps.backbones_cnn.ConvNeXtTinyBackboneOLD(
            contiguous_after_permute=True
        )

    if backbone_class == "ConvNeXtTinyBackbone":
        backbone = src.comps.backbones_cnn.ConvNeXtTinyBackbone(
            backbone_img_size
        )
    
    if backbone_class == "ConvNeXtTinyPyramidBackbone":
        backbone = src.comps.backbones_cnn_pyramid.ConvNeXtTinyPyramidBackbone(
            backbone_img_size
        )

    if backbone_class == "ConvNeXtV2TinyMultilevelBackbone":
        backbone = src.comps.backbones_cnn_pyramid.ConvNeXtV2TinyMultilevelBackbone(
            backbone_img_size
        )

    # CvT
    
    if backbone_class == "CvT21Backbone":
        backbone = src.comps.backbones_trf.CvT21Backbone(
            img_size=backbone_img_size
        )

    # Swin Transformer V2
    
    if backbone_class == "SwinTransformerV2TinyBackbone":
        backbone = src.comps.backbones_trf.SwinTransformerV2TinyBackbone(
            backbone_img_size
        )

    # GCVit
    
    if backbone_class == "GCVitTinyBackbone":
        backbone = src.comps.backbones_trf.GCVitTinyBackbone(
            img_size=backbone_img_size
        )

    # Faster-Vit
    
    if backbone_class == "FasterVit0Backbone":
        backbone = src.comps.backbones_trf.FasterVit0Backbone(
            img_size=backbone_img_size
        )

    return backbone



def create_head(
        backbone,
        head_options: dict
    ):
    """
    Creates a head for a MTL model.

    :param backbone: torch.nn.module
        Model backbone.
    :param head_options: dict
        Options to create the head with.

    :return: torch.nn.module
        The created head.
    """

    head_class = head_options["class"]
    
    # Basic retrieval head

    if head_class == "RetHead":
        head = src.comps.heads.RetHead(
            in_feat_shape=backbone.feature_shape,
            emb_size=head_options["emb_size"]
        )

    if head_class == "RetHeadNoFC":
        head = src.comps.heads.RetHeadNoFC(
            in_feat_shape=backbone.feature_shape
        )

    # Basic retrieval pyramid head

    if head_class == "RetrievalPyramidHead":
        head = src.comps.heads_pyramid.RetrievalPyramidHead(
            in_feat_idxs=head_options["in_feat_idxs"],
            feat_shapes=backbone.feature_shapes,
            emb_size=head_options["emb_size"]
        )

    # Retrieval pyramid heads V2

    if head_class == "RetrievalHeadPyramidBottomUpInstantSimple":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidBottomUpInstantSimple(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_size=head_options["emb_size"]
        )

    if head_class == "RetrievalHeadPyramidBottomUpProgressiveSimple":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidBottomUpProgressiveSimple(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_sizes=head_options["emb_sizes"]
        )

    if head_class == "RetrievalHeadPyramidBottomUpInstantConv":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidBottomUpInstantConv(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_size=head_options["emb_size"],
            conv_par_perc=head_options.get("conv_par_perc", 1)
        )

    if head_class == "RetrievalHeadPyramidBottomUpProgressiveConv":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidBottomUpProgressiveConv(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_sizes=head_options["emb_sizes"],
            conv_par_perc=head_options.get("conv_par_perc", 1)
        )

    if head_class == "RetrievalHeadPyramidTopDownInstantSimple":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidTopDownInstantSimple(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_size=head_options["emb_size"]
        )

    if head_class == "RetrievalHeadPyramidTopDownProgressiveSimple":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidTopDownProgressiveSimple(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_sizes=head_options["emb_sizes"]
        )

    if head_class == "RetrievalHeadPyramidTopDownInstantConv":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidTopDownInstantConv(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_size=head_options["emb_size"],
            conv_par_perc=head_options.get("conv_par_perc", 1)
        )

    if head_class == "RetrievalHeadPyramidTopDownProgressiveConv":
        head = src.comps.heads_pyramid_2.RetrievalHeadPyramidTopDownProgressiveConv(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_sizes=head_options["emb_sizes"],
            conv_par_perc=head_options.get("conv_par_perc", 1)
        )

    # Basic GLAM retrieval head

    if head_class == "RetrievalHeadGLAM":
        head = src.comps.heads_glam.RetrievalHeadGLAM(
            in_feat_shape=backbone.feature_shape,
            emb_size=head_options["emb_size"],
            glam_int_channels=head_options["glam_int_channels"],
            glam_1d_kernel_size=head_options["glam_1d_kernel_size"]
        )

    # Fusion GLAM retrieval head

    if head_class == "RetrievalGLAMHeadPyramidTopDownInstantSimple":
        head = src.comps.heads_glam.RetrievalGLAMHeadPyramidTopDownInstantSimple(
            feat_shapes=backbone.feature_shapes,
            in_feat_idxs=head_options["in_feat_idxs"],
            emb_size=head_options["emb_size"],
            glam_int_channels_list=head_options["glam_int_channels_list"],
            glam_1d_kernel_size=head_options.get("glam_1d_kernel_size", 3),
            conv1_groups=head_options.get("conv1_groups", 1)
        )

    return head



def create_optimizer(
        optimizer_params,
        optimizer_options,
        batch_size,
        num_devices,
        grad_acc_iters
        ):
    """
    Creates an optimizer for a model.
    The optimizer initial learning rate is corrected by multiplying by the actual batch size.
    
    :param optimizer_params: iterable of torch.nn.Parameter
        Parameters to update with the optimizer.
    :param optimizer_options: dict
        Options to create the optimizer with.
    :param batch_size: int
        Loading batch size during training.
    :param num_devices: int
        Number of GPUs used with PyTorch DDP during training.
    :param grad_acc_iters: int
        Number of gradient accumulation iterations during training.

    :return: torch.optim.Optimizer
        The created optimizer.
    """

    optimizer_class = optimizer_options["class"]
    
    if optimizer_class == "Adam":
        lr = optimizer_options["lr"] * batch_size * num_devices * grad_acc_iters
        optimizer = torch.optim.Adam(optimizer_params, lr)

    if optimizer_class == "SGD":
        lr = optimizer_options["lr"] * batch_size * num_devices * grad_acc_iters
        momentum = optimizer_options["momentum"]
        optimizer = torch.optim.SGD(optimizer_params, lr, momentum)

    return optimizer



def create_scheduler(
        optimizer,
        scheduler_options
        ):
    """
    Creates a learning rate scheduler for a model.
    
    :param optimizer: torch.optim.Optimizer
        Optimizer to which update the learning rate.
    :param scheduler_options: dict
        Options to create the scheduler with.

    :return: torch.optim.lr_scheduler.LRScheduler
        The created learning rate scheduler.
    """

    scheduler_class = scheduler_options["class"]

    if scheduler_class == "ExponentialLR":
        gamma = scheduler_options["gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    return scheduler



def create_early_stopper(
        early_stopper_options
        ):
    """
    Creates an early stopper for a training run.
    
    :param early_stopper_options: dict
        Options to create the early stopper with.

    :return: src.utils.train.EarlyStopper
        The created early stopper.
    """

    patience = early_stopper_options["patience"]
    min_delta = early_stopper_options["min_delta"]

    early_stopper = src.utils.train.EarlyStopper(patience, min_delta)

    return early_stopper



def create_best_tracker():
    """
    Creates a best tracker for a training run.

    :return: src.utils.train.BestTracker
        The created best tracker.
    """
    early_stopper = src.utils.train.BestTracker(0)

    return early_stopper
