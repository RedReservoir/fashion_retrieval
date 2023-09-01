import os
import shutil
import sys
import pathlib
import pickle as pkl
import json
import argparse

import numpy as np
import pandas as pd

import torch
import torchvision

from torch.utils.data import DataLoader, Subset

from datasets import deep_fashion
from arch import backbones, models, heads

from tqdm import tqdm

import utils.mem
import utils.list
import utils.train
import utils.time
import utils.log
import utils.dict
import utils.sig
import utils.pkl

from time import time
from datetime import datetime

from itertools import chain
from functools import reduce

import json
import socket

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



########
# COMPONENT FUNCTIONS
########



def create_backbone(backbone_class):

    if backbone_class == "ResNet50Backbone":
        backbone = backbones.ResNet50Backbone()
    if backbone_class == "EfficientNetB3Backbone":
        backbone = backbones.EfficientNetB3Backbone()
    if backbone_class == "EfficientNetB4Backbone":
        backbone = backbones.EfficientNetB4Backbone(batchnorm_eps=1e-4)
    if backbone_class == "EfficientNetB5Backbone":
        backbone = backbones.EfficientNetB5Backbone(batchnorm_eps=1e-4)
    if backbone_class == "ConvNeXtTinyBackbone":
        backbone = backbones.ConvNeXtTinyBackbone(contiguous_after_permute=True)

    return backbone


def save_experiment_checkpoint(
        experiment_checkpoint_filename,
        backbone,
        ret_head,
        optimizer,
        scheduler,
        early_stopper,
        best_tracker
        ):

    experiment_checkpoint = {
        "backbone_state_dict": backbone.state_dict(),
        "ret_head_state_dict": ret_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "early_stopper_state_dict": early_stopper.state_dict(),
        "best_tracker_state_dict": best_tracker.state_dict()
        }

    torch.save(experiment_checkpoint, experiment_checkpoint_filename)


def load_stage_1_experiment_checkpoint(
        stage_1_experiment_checkpoint_filename,
        exp_params,
        device
        ):

    # Load checkpoint

    experiment_checkpoint = torch.load(stage_1_experiment_checkpoint_filename)

    # Backbone

    backbone_class = exp_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class).to(device)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(device)

    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    # Optimizer

    optimizer_params = ret_head.parameters()

    num_devices = len(exp_params["settings"]["device_idxs"])
    optimizer_class = exp_params["settings"]["stage_1"]["optimizer"]["class"]
    
    if optimizer_class == "Adam":
        optimizer = torch.optim.Adam(optimizer_params)
    if optimizer_class == "SGD":
        lr = exp_params["settings"]["stage_1"]["optimizer"]["lr"] * num_devices
        optimizer = torch.optim.SGD(optimizer_params, lr)

    optimizer.load_state_dict(experiment_checkpoint["optimizer_state_dict"])

    # Scheduler

    scheduler_class = exp_params["settings"]["stage_1"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":
        gamma = exp_params["settings"]["stage_1"]["scheduler"]["gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    scheduler.load_state_dict(experiment_checkpoint["scheduler_state_dict"])

    # Early stopper

    early_stopper = utils.train.EarlyStopper()

    early_stopper.load_state_dict(experiment_checkpoint["early_stopper_state_dict"])

    # Best tracker

    best_tracker = utils.train.BestTracker()

    best_tracker.load_state_dict(experiment_checkpoint["best_tracker_state_dict"])

    return (
        backbone,
        ret_head,
        optimizer,
        scheduler,
        early_stopper,
        best_tracker
    )


def load_stage_2_experiment_checkpoint(
        stage_2_experiment_checkpoint_filename,
        exp_params,
        device
        ):

    # Load checkpoint

    experiment_checkpoint = torch.load(stage_2_experiment_checkpoint_filename)

    # Backbone

    backbone_class = exp_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class).to(device)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(device)

    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    # Optimizer

    optimizer_params = chain(
        backbone.parameters(),
        ret_head.parameters()
    )

    num_devices = len(exp_params["settings"]["device_idxs"])
    optimizer_class = exp_params["settings"]["stage_2"]["optimizer"]["class"]
    
    if optimizer_class == "Adam":
        optimizer = torch.optim.Adam(optimizer_params)
    if optimizer_class == "SGD":
        lr = exp_params["settings"]["stage_2"]["optimizer"]["lr"] * num_devices
        optimizer = torch.optim.SGD(optimizer_params, lr)

    optimizer.load_state_dict(experiment_checkpoint["optimizer_state_dict"])

    # Scheduler

    scheduler_class = exp_params["settings"]["stage_2"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = exp_params["settings"]["stage_1"]["scheduler"]["gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    scheduler.load_state_dict(experiment_checkpoint["scheduler_state_dict"])

    # Early stopper

    early_stopper = utils.train.EarlyStopper()

    early_stopper.load_state_dict(experiment_checkpoint["early_stopper_state_dict"])

    # Best tracker

    best_tracker = utils.train.BestTracker()

    best_tracker.load_state_dict(experiment_checkpoint["best_tracker_state_dict"])

    return (
        backbone,
        ret_head,
        optimizer,
        scheduler,
        early_stopper,
        best_tracker
    )


def initialize_stage_1_components(exp_params, device):

    # Backbone

    backbone_class = exp_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class).to(device)

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(device)
    
    # Optimizer

    optimizer_params = ret_head.parameters()

    num_devices = len(exp_params["settings"]["device_idxs"])
    optimizer_class = exp_params["settings"]["stage_1"]["optimizer"]["class"]

    if optimizer_class == "Adam":

        lr = exp_params["settings"]["stage_1"]["optimizer"]["lr"] * num_devices

        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=lr
        )

    if optimizer_class == "SGD":

        lr = exp_params["settings"]["stage_1"]["optimizer"]["lr"] * num_devices
        momentum = exp_params["settings"]["stage_1"]["optimizer"]["momentum"]

        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=lr,
            momentum=momentum
        )

    # Scheduler

    scheduler_class = exp_params["settings"]["stage_1"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = exp_params["settings"]["stage_1"]["scheduler"]["gamma"]

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

    # Early stopper

    patience = exp_params["settings"]["stage_1"]["early_stopper"]["patience"]
    min_delta = exp_params["settings"]["stage_1"]["early_stopper"]["min_delta"]

    early_stopper = utils.train.EarlyStopper(
        patience=patience,
        min_delta=min_delta
    )

    # Best tracker

    best_tracker = utils.train.BestTracker()

    return (
        backbone,
        ret_head,
        optimizer,
        scheduler,
        early_stopper,
        best_tracker
    )


def initialize_stage_2_components(
        best_stage_1_experiment_checkpoint_filename,
        exp_params,
        device
        ):

    # Load checkpoint

    experiment_checkpoint = torch.load(best_stage_1_experiment_checkpoint_filename)

    # Backbone

    backbone_class = exp_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class).to(device)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(device)

    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    # Optimizer

    optimizer_params = chain(
        backbone.parameters(),
        ret_head.parameters()
    )

    num_devices = len(exp_params["settings"]["device_idxs"])
    optimizer_class = exp_params["settings"]["stage_2"]["optimizer"]["class"]

    if optimizer_class == "Adam":

        lr = exp_params["settings"]["stage_2"]["optimizer"]["lr"] * num_devices

        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=lr
        )

    if optimizer_class == "SGD":

        lr = exp_params["settings"]["stage_2"]["optimizer"]["lr"] * num_devices
        momentum = exp_params["settings"]["stage_2"]["optimizer"]["momentum"]

        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=lr,
            momentum=momentum
        )

    # Scheduler

    scheduler_class = exp_params["settings"]["stage_2"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = exp_params["settings"]["stage_2"]["scheduler"]["gamma"]

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

    # Early stopper

    patience = exp_params["settings"]["stage_2"]["early_stopper"]["patience"]
    min_delta = exp_params["settings"]["stage_2"]["early_stopper"]["min_delta"]

    early_stopper = utils.train.EarlyStopper(
        patience=patience,
        min_delta=min_delta
    )

    # Best tracker

    best_tracker = utils.train.BestTracker()

    return (
        backbone,
        ret_head,
        optimizer,
        scheduler,
        early_stopper,
        best_tracker
    )



########
# EXPERIMENT DATA FUNCTIONS
########



def save_exp_data(
        exp_data_filename,
        exp_data
        ):

    with open(exp_data_filename, 'w') as exp_data_file:
        json.dump(exp_data, exp_data_file, indent=2)


def load_exp_data(
        exp_data_filename
        ):

    with open(exp_data_filename, 'r') as exp_data_file:
        exp_data = json.load(exp_data_file)

    return exp_data


def initialize_stage_1_exp_data(
        exp_data,
        exp_params
        ):

    # Settings

    exp_data["settings"]["stage_1"] = {}
    exp_data["settings"]["stage_1"].update(exp_params["settings"]["stage_1"])
    
    stage_description = "Train with frozen backbone"
    exp_data["settings"]["stage_1"]["description"] = stage_description

    exp_data["settings"]["stage_1"]["learning_rate_list"] = []
   
    exp_data["settings"]["stage_1"]["extra"] = [
        "automatic_mixed_precision"
    ]

    # Results

    exp_data["results"]["stage_1"] = {}

    exp_data["results"]["stage_1"]["train_epoch_time_list"] = []
    exp_data["results"]["stage_1"]["val_epoch_time_list"] = []

    exp_data["results"]["stage_1"]["train_mean_loss_list"] = []
    exp_data["results"]["stage_1"]["val_mean_loss_list"] = []

    exp_data["results"]["stage_1"]["num_epochs"] = 0
    exp_data["results"]["stage_1"]["finished"] = False

    return exp_data


def initialize_stage_2_exp_data(exp_data, exp_params):

    # Settings

    exp_data["settings"]["stage_2"] = {}
    exp_data["settings"]["stage_2"].update(exp_params["settings"]["stage_2"])

    stage_description = "Train entire model"
    exp_data["settings"]["stage_2"]["description"] = stage_description
        
    exp_data["settings"]["stage_2"]["learning_rate_list"] = []

    exp_data["settings"]["stage_2"]["extra"] = [
        "automatic_mixed_precision"
    ]
    
    # Results

    exp_data["results"]["stage_2"] = {}

    exp_data["results"]["stage_2"]["train_epoch_time_list"] = []
    exp_data["results"]["stage_2"]["val_epoch_time_list"] = []

    exp_data["results"]["stage_2"]["train_mean_loss_list"] = []
    exp_data["results"]["stage_2"]["val_mean_loss_list"] = []

    exp_data["results"]["stage_2"]["num_epochs"] = 0
    exp_data["results"]["stage_2"]["finished"] = False

    return exp_data


def initialize_test_exp_data(
        exp_data,
        exp_params
        ):

    # Settings

    exp_data["settings"]["test"] = {}
    exp_data["settings"]["test"].update(exp_params["settings"]["test"])
    
    stage_description = "Test final model performance"
    exp_data["settings"]["test"]["description"] = stage_description
    
    # Results

    exp_data["results"]["test"] = {}

    exp_data["results"]["test"]["finished"] = False

    return exp_data



########
# TRAIN & EVAL FUNCTIONS
########



def train_epoch(
        data_loader,
        ret_model,
        optimizer,
        scaler,
        device,
        logger,
        max_acc_iter=1,
        with_tqdm=True
    ):

    ret_model.train()

    total_loss_item = 0
    curr_acc_iter = 0

    loader_gen = data_loader
    if with_tqdm: loader_gen = tqdm(loader_gen)

    for batch in loader_gen:
            
        batch_loss = batch_evaluation(batch, ret_model, device, logger)
        total_loss_item += batch_loss.sum().item()

        scaler.scale(batch_loss.mean() / max_acc_iter).backward()
        curr_acc_iter += 1

        if curr_acc_iter == max_acc_iter:

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            curr_acc_iter = 0

    if curr_acc_iter > 0:

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss_item


def eval_epoch(
        data_loader,
        ret_model,
        device,
        logger,
        with_tqdm=True
    ):

    ret_model.eval()
    
    total_loss_item = 0

    with torch.no_grad():

        loader_gen = data_loader
        if with_tqdm: loader_gen = tqdm(loader_gen)

        for batch in loader_gen:

            batch_loss = batch_evaluation(batch, ret_model, device, logger)
            total_loss_item += batch_loss.sum().item()

    return total_loss_item


def batch_evaluation(
        batch,
        ret_model,
        device,
        logger
    ):

    anc_imgs = batch[0].to(device)
    pos_imgs = batch[1].to(device)
    neg_imgs = batch[2].to(device)

    with torch.cuda.amp.autocast():

        anc_emb = ret_model(anc_imgs)
        pos_emb = ret_model(pos_imgs)
        neg_emb = ret_model(neg_imgs)

        triplet_loss = torch.nn.TripletMarginLoss(reduction="none")
        batch_loss = triplet_loss(anc_emb, pos_emb, neg_emb)

    if np.isnan(batch_loss.sum().item()):

        logger.print("WARNING: Batch produced nan loss")
        logger.print("  batch length", batch_loss.size(dim=0))
        logger.print("  batch_loss nans", torch.isnan(batch_loss).any())
        logger.print("  anc_imgs nans", torch.isnan(anc_imgs).any())
        logger.print("  pos_imgs nans", torch.isnan(pos_imgs).any())
        logger.print("  neg_imgs nans", torch.isnan(neg_imgs).any())
        logger.print("  anc_emb nans", torch.isnan(anc_emb).any())
        logger.print("  pos_emb nans", torch.isnan(pos_emb).any())
        logger.print("  neg_emb nans", torch.isnan(neg_emb).any())

    return batch_loss



########
# STAGES
########



def execute_stage_1_DDP(
    rank,
    world_size,
    command_args,
    exp_params,
    exp_data,
    train_epoch_time_list_mp,
    val_epoch_time_list_mp,
    train_mean_loss_list_mp,
    val_mean_loss_list_mp,
    finished_mp
):

    # Prepare logger

    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "exp_logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.silent: logger_streams.append(sys.stdout)

    logger = utils.log.Logger(logger_streams)

    # PyTorch DDP stuff

    logger.print("  [Rank {:d}] DDP Setup Preparing...".format(rank))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(rank)
    
    logger.print("  [Rank {:d}] DDP Setup Ready".format(rank))

    # Dataset initialization

    logger.print("  [Rank {:d}] Data Preparing...".format(rank))

    backbone_class = exp_params["settings"]["backbone"]["class"]

    if backbone_class == "ResNet50Backbone":
        backbone_image_transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB3Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB4Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B4_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB5Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B5_Weights.DEFAULT.transforms()
    if backbone_class == "ConvNeXtTinyBackbone":
        backbone_image_transform = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()

    backbone_image_transform.antialias = True

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")

    ctsrbm_dataset = deep_fashion.ConsToShopClothRetrBM_NEW(ctsrbm_dataset_dir, img_transform=backbone_image_transform, neg_img_filename_list_id="test")

    train_idxs = ctsrbm_dataset.get_split_mask_idxs("train")
    val_idxs = ctsrbm_dataset.get_split_mask_idxs("val")

    cutdown_ratio = 1
    if cutdown_ratio != 1:

        train_idxs = utils.list.cutdown_list(train_idxs, cutdown_ratio)
        val_idxs = utils.list.cutdown_list(val_idxs, cutdown_ratio)
        
        logger.print("  [Rank {:d}] Reducing dataset to {:5.2f}%".format(
            rank,
            cutdown_ratio * 100
        ))

    ctsrbm_train_dataset = Subset(ctsrbm_dataset, train_idxs)
    ctsrbm_val_dataset = Subset(ctsrbm_dataset, val_idxs)

    # Data loader initialization

    batch_size = exp_params["settings"]["stage_1"]["data_loading"]["batch_size"]
    num_workers = exp_params["settings"]["stage_1"]["data_loading"]["num_workers"]

    train_loader = DataLoader(
        ctsrbm_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_train_dataset)
    )
    
    val_loader = DataLoader(
        ctsrbm_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_val_dataset)
    )
    
    logger.print("  [Rank {:d}] Data Prepared".format(rank))

    # Settings & components

    logger.print("  [Rank {:d}] Components Loading...".format(rank))

    ## Basic settings

    max_epochs = exp_params["settings"]["stage_1"]["max_epochs"]
    num_epochs = exp_data["results"]["stage_1"]["num_epochs"]
    
    ## Load or initialize components

    if num_epochs == 0:

        backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
        initialize_stage_1_components(
            exp_params,
            device
        )

    else:

        last_stage_1_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "last_stage_1_ckp.pth"
        )

        backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
        load_stage_1_experiment_checkpoint(
            last_stage_1_experiment_checkpoint_filename,
            exp_params,
            device
        )

    ## Build models

    ret_model = models.BackboneAndHead(backbone, ret_head).to(device)
    ret_model = DDP(ret_model, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)

    ## General settings

    with_tqdm = not command_args.notqdm and not command_args.silent

    max_acc_iter = utils.dict.chain_get(
        exp_params,
        "settings", "stage_1", "max_acc_iter",
        default=1
    )

    scaler = torch.cuda.amp.GradScaler()

    for param in backbone.parameters():
        param.requires_grad = False      

    logger.print("  [Rank {:d}] Components Loaded".format(rank))

    # Training loop

    logger.print("  [Rank {:d}] Training Loop Begin".format(rank))

    if rank == 0:    
        finished_mp.value = exp_data["results"]["stage_1"]["finished"]

    torch.distributed.barrier()

    while not finished_mp.value:

        ## Epoch pre-processing

        num_epochs += 1

        logger.print("  [Rank {:d}] Entering epoch {:d}".format(
            rank,
            num_epochs
        ))

        ## Training

        start_time = time()

        train_loss = train_epoch(
            train_loader,
            ret_model,
            optimizer,
            scaler,
            device,
            logger,
            max_acc_iter=max_acc_iter,
            with_tqdm=with_tqdm
        )

        end_time = time()

        train_epoch_time = end_time - start_time
        train_mean_loss = train_loss / len(ctsrbm_train_dataset)

        logger.print("  [Rank {:d}] Train epoch time: {:s}".format(
            rank,
            utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        logger.print("  [Rank {:d}] Train mean loss:  {:.2e}".format(
            rank,
            train_mean_loss
        ))

        ## Validation

        start_time = time()

        val_loss = eval_epoch(
            val_loader,
            ret_model,
            device,
            logger,
            with_tqdm=with_tqdm
        )

        end_time = time()

        val_epoch_time = end_time - start_time
        val_mean_loss = val_loss / len(ctsrbm_val_dataset)

        logger.print("  [Rank {:d}] Val epoch time:   {:s}".format(
            rank,
            utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        logger.print("  [Rank {:d}] Val mean loss:    {:.2e}".format(
            rank,
            train_mean_loss
        ))

        scheduler.step()

        if rank == 0:
            logger.print("  [Rank {:d}] Current memory usage:".format(rank))
            logger.print(utils.mem.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=4))

        ## Track results

        train_epoch_time_list_mp[rank] = train_epoch_time
        val_epoch_time_list_mp[rank] = val_epoch_time
        train_mean_loss_list_mp[rank] = train_mean_loss
        val_mean_loss_list_mp[rank] = val_mean_loss

        torch.distributed.barrier()

        if rank == 0:

            exp_data["results"]["stage_1"]["train_epoch_time_list"].append(list(train_epoch_time_list_mp))
            exp_data["results"]["stage_1"]["val_epoch_time_list"].append(list(val_epoch_time_list_mp))

            exp_data["results"]["stage_1"]["train_mean_loss_list"].append(list(train_mean_loss_list_mp))
            exp_data["results"]["stage_1"]["val_mean_loss_list"].append(list(val_mean_loss_list_mp))

            exp_data["settings"]["stage_1"]["learning_rate_list"].append(scheduler.get_last_lr()[0])
        
        ## Training conditions and checkpoints

        if rank == 0:

            ## Number of epochs

            if num_epochs >= max_epochs:
                finished_mp.value = True

            ## Early stopping

            if early_stopper.early_stop(sum(train_mean_loss_list_mp)):
                finished_mp.value = True

            ## Checkpoint saving

            with utils.sig.DelayedInterrupt():

                ## Save best component checkpoint

                if best_tracker.is_best(sum(train_mean_loss_list_mp)):

                    best_stage_1_experiment_checkpoint_filename = os.path.join(
                        experiment_dirname, "best_stage_1_ckp.pth"
                    )

                    logger.print("  [Rank {:d}] Saving currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_1_experiment_checkpoint_filename
                    ))

                    save_experiment_checkpoint(
                        best_stage_1_experiment_checkpoint_filename,
                        backbone,
                        ret_head,
                        optimizer,
                        scheduler,
                        early_stopper,
                        best_tracker
                        )

                    logger.print("  [Rank {:d}] Saved currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_1_experiment_checkpoint_filename
                    ))
                    
                ## Save experiment data

                exp_data["results"]["stage_1"]["num_epochs"] = num_epochs
                exp_data["results"]["stage_1"]["finished"] = finished_mp.value

                exp_data_filename = os.path.join(
                    experiment_dirname, "exp_data.json"
                )

                logger.print("  [Rank {:d}] Saving experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                save_exp_data(
                    exp_data_filename, exp_data
                )

                logger.print("  [Rank {:d}] Saved experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                ## Save last component checkpoint
                
                last_stage_1_experiment_checkpoint_filename = os.path.join(
                    experiment_dirname, "last_stage_1_ckp.pth"
                )

                logger.print("  [Rank {:d}] Saving last epoch model to \"{:s}\"".format(
                    rank,
                    last_stage_1_experiment_checkpoint_filename
                ))

                save_experiment_checkpoint(
                    last_stage_1_experiment_checkpoint_filename,
                    backbone,
                    ret_head,
                    optimizer,
                    scheduler,
                    early_stopper,
                    best_tracker
                )

                logger.print("  [Rank {:d}] Saved last epoch model to \"{:s}\"".format(
                    rank,
                    last_stage_1_experiment_checkpoint_filename
                ))

        torch.distributed.barrier()

    logger.print("  [Rank {:d}] Training Loop End".format(rank))

    # PyTorch DDP stuff

    destroy_process_group()


def execute_stage_2_DDP(
    rank,
    world_size,
    command_args,
    exp_params,
    exp_data,
    train_epoch_time_list_mp,
    val_epoch_time_list_mp,
    train_mean_loss_list_mp,
    val_mean_loss_list_mp,
    finished_mp
):

    # Prepare logger

    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "exp_logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.silent: logger_streams.append(sys.stdout)

    logger = utils.log.Logger(logger_streams)

    logger.print("  [Rank {:d}] Logger prepared".format(rank))

    # PyTorch DDP stuff

    logger.print("  [Rank {:d}] DDP Setup Preparing...".format(rank))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(rank)
    
    logger.print("  [Rank {:d}] DDP Setup Ready".format(rank))

    # Dataset initialization

    logger.print("  [Rank {:d}] Data Preparing...".format(rank))

    backbone_class = exp_params["settings"]["backbone"]["class"]

    if backbone_class == "ResNet50Backbone":
        backbone_image_transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB3Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB4Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B4_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB5Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B5_Weights.DEFAULT.transforms()
    if backbone_class == "ConvNeXtTinyBackbone":
        backbone_image_transform = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()

    backbone_image_transform.antialias = True

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")

    ctsrbm_dataset = deep_fashion.ConsToShopClothRetrBM_NEW(ctsrbm_dataset_dir, img_transform=backbone_image_transform, neg_img_filename_list_id="test")

    train_idxs = ctsrbm_dataset.get_split_mask_idxs("train")
    val_idxs = ctsrbm_dataset.get_split_mask_idxs("val")

    cutdown_ratio = 1
    if cutdown_ratio != 1:

        train_idxs = utils.list.cutdown_list(train_idxs, cutdown_ratio)
        val_idxs = utils.list.cutdown_list(val_idxs, cutdown_ratio)

        logger.print("  [Rank {:d}] Reducing dataset to {:5.2f}%".format(
            rank,
            cutdown_ratio * 100
        ))
    
    ctsrbm_train_dataset = Subset(ctsrbm_dataset, train_idxs)
    ctsrbm_val_dataset = Subset(ctsrbm_dataset, val_idxs)

    # Data loader initialization

    batch_size = exp_params["settings"]["stage_2"]["data_loading"]["batch_size"]
    num_workers = exp_params["settings"]["stage_2"]["data_loading"]["num_workers"]

    train_loader = DataLoader(
        ctsrbm_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_train_dataset)
    )
    
    val_loader = DataLoader(
        ctsrbm_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_val_dataset)
    )
    
    logger.print("  [Rank {:d}] Data Prepared".format(rank))

    # Settings & components
    
    logger.print("  [Rank {:d}] Components Loading...".format(rank))

    ## Basic settings

    max_epochs = exp_params["settings"]["stage_2"]["max_epochs"]
    num_epochs = exp_data["results"]["stage_2"]["num_epochs"]
    
    ## Load or initialize components

    if num_epochs == 0:

        best_stage_1_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "best_stage_1_ckp.pth"
        )

        backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
        initialize_stage_2_components(
            best_stage_1_experiment_checkpoint_filename,
            exp_params,
            device
        )

    else:

        last_stage_2_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "last_stage_2_ckp.pth"
        )

        backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
        load_stage_2_experiment_checkpoint(
            last_stage_2_experiment_checkpoint_filename,
            exp_params,
            device
        )

    ## Build models

    ret_model = models.BackboneAndHead(backbone, ret_head).to(device)
    ret_model = DDP(ret_model, device_ids=[rank], broadcast_buffers=False)

    ## General settings

    with_tqdm = not command_args.notqdm and not command_args.silent

    max_acc_iter = utils.dict.chain_get(
        exp_params,
        "settings", "stage_2", "max_acc_iter",
        default=1
    )

    scaler = torch.cuda.amp.GradScaler()

    for param in backbone.parameters():
        param.requires_grad = True      

    logger.print("  [Rank {:d}] Components Loaded".format(rank))

    # Training loop

    logger.print("  [Rank {:d}] Training Loop Begin".format(rank))

    if rank == 0:    
        finished_mp.value = exp_data["results"]["stage_2"]["finished"]

    torch.distributed.barrier()

    while not finished_mp.value:

        ## Epoch pre-processing

        num_epochs += 1

        logger.print("  [Rank {:d}] Entering epoch {:d}".format(
            rank,
            num_epochs
        ))

        ## Training

        start_time = time()

        train_loss = train_epoch(
            train_loader,
            ret_model,
            optimizer,
            scaler,
            device,
            logger,
            max_acc_iter=max_acc_iter,
            with_tqdm=with_tqdm
        )

        end_time = time()

        train_epoch_time = end_time - start_time
        train_mean_loss = train_loss / len(ctsrbm_train_dataset)

        logger.print("  [Rank {:d}] Train epoch time: {:s}".format(
            rank,
            utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        logger.print("  [Rank {:d}] Train mean loss:  {:.2e}".format(
            rank,
            train_mean_loss
        ))

        ## Validation

        start_time = time()

        val_loss = eval_epoch(
            val_loader,
            ret_model,
            device,
            logger,
            with_tqdm=with_tqdm
        )

        end_time = time()

        val_epoch_time = end_time - start_time
        val_mean_loss = val_loss / len(ctsrbm_val_dataset)

        logger.print("  [Rank {:d}] Val epoch time:   {:s}".format(
            rank,
            utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        logger.print("  [Rank {:d}] Val mean loss:    {:.2e}".format(
            rank,
            train_mean_loss
        ))

        scheduler.step()

        if rank == 0:
            logger.print("  [Rank {:d}] Current memory usage:".format(rank))
            logger.print(utils.mem.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=4))

        ## Track results

        train_epoch_time_list_mp[rank] = train_epoch_time
        val_epoch_time_list_mp[rank] = val_epoch_time
        train_mean_loss_list_mp[rank] = train_mean_loss
        val_mean_loss_list_mp[rank] = val_mean_loss

        torch.distributed.barrier()

        if rank == 0:

            exp_data["results"]["stage_2"]["train_epoch_time_list"].append(list(train_epoch_time_list_mp))
            exp_data["results"]["stage_2"]["val_epoch_time_list"].append(list(val_epoch_time_list_mp))

            exp_data["results"]["stage_2"]["train_mean_loss_list"].append(list(train_mean_loss_list_mp))
            exp_data["results"]["stage_2"]["val_mean_loss_list"].append(list(val_mean_loss_list_mp))

            exp_data["settings"]["stage_2"]["learning_rate_list"].append(scheduler.get_last_lr()[0])
        
        ## Training conditions and checkpoints

        if rank == 0:

            ## Number of epochs

            if num_epochs >= max_epochs:
                finished_mp.value = True

            ## Early stopping

            if early_stopper.early_stop(sum(train_mean_loss_list_mp)):
                finished_mp.value = True

            ## Checkpoint saving

            with utils.sig.DelayedInterrupt():

                ## Save best component checkpoint

                if best_tracker.is_best(sum(train_mean_loss_list_mp)):

                    best_stage_2_experiment_checkpoint_filename = os.path.join(
                        experiment_dirname, "best_stage_2_ckp.pth"
                    )

                    logger.print("  [Rank {:d}] Saving currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_2_experiment_checkpoint_filename
                    ))

                    save_experiment_checkpoint(
                        best_stage_2_experiment_checkpoint_filename,
                        backbone,
                        ret_head,
                        optimizer,
                        scheduler,
                        early_stopper,
                        best_tracker
                        )

                    logger.print("  [Rank {:d}] Saved currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_2_experiment_checkpoint_filename
                    ))
                    
                ## Save experiment data

                exp_data["results"]["stage_2"]["num_epochs"] = num_epochs
                exp_data["results"]["stage_2"]["finished"] = finished_mp.value

                exp_data_filename = os.path.join(
                    experiment_dirname, "exp_data.json"
                )

                logger.print("  [Rank {:d}] Saving experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                save_exp_data(
                    exp_data_filename, exp_data
                )

                logger.print("  [Rank {:d}] Saved experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                ## Save last component checkpoint
                
                last_stage_2_experiment_checkpoint_filename = os.path.join(
                    experiment_dirname, "last_stage_2_ckp.pth"
                )

                logger.print("  [Rank {:d}] Saving last epoch model to \"{:s}\"".format(
                    rank,
                    last_stage_2_experiment_checkpoint_filename
                ))

                save_experiment_checkpoint(
                    last_stage_2_experiment_checkpoint_filename,
                    backbone,
                    ret_head,
                    optimizer,
                    scheduler,
                    early_stopper,
                    best_tracker
                )

                logger.print("  [Rank {:d}] Saved last epoch model to \"{:s}\"".format(
                    rank,
                    last_stage_2_experiment_checkpoint_filename
                ))

        torch.distributed.barrier()

    logger.print("  [Rank {:d}] Training Loop End".format(rank))

    # PyTorch DDP stuff

    destroy_process_group()


def execute_test_DDP(
    rank,
    world_size,
    command_args,
    exp_params,
    exp_data,
    test_epoch_time_list_mp,
    test_mean_loss_list_mp,
    finished_mp
):

    # Prepare logger

    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "exp_logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.silent: logger_streams.append(sys.stdout)

    logger = utils.log.Logger(logger_streams)

    logger.print("  [Rank {:d}] Logger prepared".format(rank))

    # PyTorch DDP stuff

    logger.print("  [Rank {:d}] DDP Setup Preparing...".format(rank))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(rank)
    
    logger.print("  [Rank {:d}] DDP Setup Ready".format(rank))

    # Dataset initialization

    logger.print("  [Rank {:d}] Data Preparing...".format(rank))
    
    backbone_class = exp_params["settings"]["backbone"]["class"]

    if backbone_class == "ResNet50Backbone":
        backbone_image_transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB3Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB4Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B4_Weights.DEFAULT.transforms()
    if backbone_class == "EfficientNetB5Backbone":
        backbone_image_transform = torchvision.models.EfficientNet_B5_Weights.DEFAULT.transforms()
    if backbone_class == "ConvNeXtTinyBackbone":
        backbone_image_transform = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()

    backbone_image_transform.antialias = True

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")

    ctsrbm_dataset = deep_fashion.ConsToShopClothRetrBM_NEW(ctsrbm_dataset_dir, img_transform=backbone_image_transform, neg_img_filename_list_id="test")

    test_idxs = ctsrbm_dataset.get_split_mask_idxs("test")

    cutdown_ratio = 1
    if cutdown_ratio != 1:

        test_idxs = utils.list.cutdown_list(test_idxs, cutdown_ratio)

        logger.print("  [Rank {:d}] Reducing dataset to {:5.2f}%".format(
            rank,
            cutdown_ratio * 100
        ))

    """
    if len(exp_params["settings"]["device_idxs"]) > 1:

        test_idxs = test_idxs[:(len(test_idxs) // batch_size) * batch_size]
    """
    
    ctsrbm_test_dataset = Subset(ctsrbm_dataset, test_idxs)

    # Data loader initialization

    batch_size = exp_params["settings"]["test"]["data_loading"]["batch_size"]
    num_workers = exp_params["settings"]["test"]["data_loading"]["num_workers"]

    test_loader = DataLoader(
        ctsrbm_test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_test_dataset)
    )
    
    logger.print("  [Rank {:d}] Data Prepared".format(rank))

    # Settings & components
    
    logger.print("  [Rank {:d}] Components Loading...".format(rank))

    ## Load or initialize components

    best_stage_2_experiment_checkpoint_filename = os.path.join(
        experiment_dirname, "best_stage_2_ckp.pth"
    )

    backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
    load_stage_2_experiment_checkpoint(
        best_stage_2_experiment_checkpoint_filename,
        exp_params,
        device
    )
        
    ## Build models

    ret_model = models.BackboneAndHead(backbone, ret_head).to(device)
    ret_model = DDP(ret_model, device_ids=[rank], broadcast_buffers=False)

    ## General settings

    with_tqdm = not command_args.notqdm and not command_args.silent

    logger.print("  [Rank {:d}] Components Loaded".format(rank))

    # Testing

    logger.print("  [Rank {:d}] Testing Begin".format(rank))

    if rank == 0:    
        finished_mp.value = exp_data["results"]["test"]["finished"]

    torch.distributed.barrier()

    if not finished_mp.value:

        ## Test epoch

        start_time = time()

        test_loss = eval_epoch(
            test_loader,
            ret_model,
            device,
            logger,
            with_tqdm=with_tqdm
        )

        end_time = time()

        test_epoch_time = end_time - start_time
        test_mean_loss = test_loss / len(ctsrbm_test_dataset)

        logger.print("  [Rank {:d}] Test epoch time:  {:s}".format(
            rank,
            utils.time.sprint_fancy_time_diff(test_epoch_time)
        ))

        logger.print("  [Rank {:d}] Test mean loss:   {:.2e}".format(
            rank,
            test_mean_loss
        ))

        if rank == 0:
            logger.print("  [Rank {:d}] Current memory usage:".format(rank))
            logger.print(utils.mem.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=4))
        
        ## Track results

        test_epoch_time_list_mp[rank] = test_epoch_time
        test_mean_loss_list_mp[rank] = test_mean_loss

        torch.distributed.barrier()

        if rank == 0:

            exp_data["results"]["test"]["train_epoch_time"] = list(test_epoch_time_list_mp)

            exp_data["results"]["test"]["test_mean_loss"] = list(test_mean_loss_list_mp)

        ## Training conditions and checkpoints

        finished_mp.value = True        

        if rank == 0:

            ## Checkpoint saving

            with utils.sig.DelayedInterrupt():

                ## Save experiment data

                exp_data["results"]["test"]["finished"] = finished_mp.value

                exp_data_filename = os.path.join(
                    experiment_dirname, "exp_data.json"
                )

                logger.print("  [Rank {:d}] Saving experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                save_exp_data(
                    exp_data_filename, exp_data
                )

                logger.print("  [Rank {:d}] Saved experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

    logger.print("  [Rank {:d}] Testing End".format(rank))

    # PyTorch DDP stuff

    destroy_process_group()



########
# MAIN SCRIPT
########



if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    datetime_now_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")


    ####
    # COMMAND ARGUMENTS
    ####


    parser = argparse.ArgumentParser()
    parser.add_argument("exp_params_filename", help="filename of the experiment params json file inside the \"exp_params\" directory")
    parser.add_argument("--silent", help="no terminal prints will be made", action="store_true")
    parser.add_argument("--notqdm", help="no tqdm bars will be shown", action="store_true")
    parser.add_argument("--reset", help="experiment directory will be reset", action="store_true")
    command_args = parser.parse_args()

    exp_params_filename = os.path.join(pathlib.Path.home(), "fashion_retrieval", "exp_params", command_args.exp_params_filename)


    ####
    # EXPERIMENT PREREQUISITES
    ####


    # Read experiment params

    exp_params_file = open(exp_params_filename, 'r')
    exp_params = json.load(exp_params_file)
    exp_params_file.close()

    # Check and create experiment directory

    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)
    
    if os.path.exists(experiment_dirname) and command_args.reset:
        shutil.rmtree(experiment_dirname)

    if not os.path.exists(experiment_dirname):
        os.mkdir(experiment_dirname)
    

    ####
    # PREPARE LOGGER
    ####


    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "exp_logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.silent: logger_streams.append(sys.stdout)

    logger = utils.log.Logger(logger_streams)
    

    ####
    # PREPARE EXPERIMENT DATA
    ####

    
    exp_data_filename = os.path.join(
        experiment_dirname, "exp_data.json"
    )

    if not os.path.exists(exp_data_filename):

        exp_data = {}
        exp_data["experiment_name"] = experiment_name
        exp_data["settings"] = {}
        exp_data["results"] = {}

        exp_data["settings"]["datasets"] = [
            "DeepFashion Consumer-to-shop Clothes Retrieval Benchmark"
        ]

        exp_data["settings"]["backbone"] = {
            "class": exp_params["settings"]["backbone"]["class"]
        }

        exp_data["settings"]["heads"] = []
        exp_data["settings"]["heads"].append({
            "class": "RetHead"
        })

        exp_data_filename = os.path.join(
            experiment_dirname, "exp_data.json"
        )
        
        save_exp_data(exp_data_filename, exp_data)

        logger.print("Starting experiment at {:s}".format(datetime_now_str))

    else:

        logger.print("Resuming experiment at {:s}".format(datetime_now_str))
    
    exp_data = load_exp_data(exp_data_filename)


    ####
    # GPU INITIALIZATION
    ####


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_idxs = exp_params["settings"]["device_idxs"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in device_idxs])

    torch.cuda.empty_cache()


    exp_data["settings"]["gpu_usage"] = utils.mem.list_gpu_data(device_idxs)
    exp_data["settings"]["hostname"] = socket.gethostname()


    ####
    # STAGE 1 - FROZEN BACKBONE
    ####


    # Initialize experiment data

    if utils.dict.chain_get(exp_data, "settings", "stage_1") is None:
        exp_data = initialize_stage_1_exp_data(exp_data, exp_params)

    # Check if stage is finished

    finished = exp_data["results"]["stage_1"]["finished"]

    if not finished:

        world_size = len(exp_params["settings"]["device_idxs"])
        manager = torch.multiprocessing.Manager()

        train_epoch_time_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        val_epoch_time_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        train_mean_loss_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        val_mean_loss_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        finished_mp = manager.Value("b", False)

        logger.print("Stage 1 - Multiprocessing start")

        torch.multiprocessing.spawn(
            execute_stage_1_DDP,
            args=[
                world_size,
                command_args,
                exp_params,
                exp_data,
                train_epoch_time_list_mp,
                val_epoch_time_list_mp,
                train_mean_loss_list_mp,
                val_mean_loss_list_mp,
                finished_mp
            ],
            nprocs=world_size
        )

    # Reload experiment data

    exp_data = load_exp_data(exp_data_filename)


    ####
    # STAGE 2 - UNFROZEN BACKBONE
    ####


    # Initialize experiment data

    if utils.dict.chain_get(exp_data, "settings", "stage_2") is None:
        exp_data = initialize_stage_2_exp_data(exp_data, exp_params)

    # Check if stage is finished

    finished = exp_data["results"]["stage_2"]["finished"]

    if not finished:

        world_size = len(exp_params["settings"]["device_idxs"])
        manager = torch.multiprocessing.Manager()

        train_epoch_time_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        val_epoch_time_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        train_mean_loss_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        val_mean_loss_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        finished_mp = manager.Value("b", False)

        logger.print("Stage 2 - Multiprocessing start")

        torch.multiprocessing.spawn(
            execute_stage_2_DDP,
            args=[
                world_size,
                command_args,
                exp_params,
                exp_data,
                train_epoch_time_list_mp,
                val_epoch_time_list_mp,
                train_mean_loss_list_mp,
                val_mean_loss_list_mp,
                finished_mp
            ],
            nprocs=world_size
        )

    # Reload experiment data

    exp_data = load_exp_data(exp_data_filename)


    ####
    # TEST
    ####


    # Initialize experiment data

    if utils.dict.chain_get(exp_data, "settings", "test") is None:
        exp_data = initialize_test_exp_data(exp_data, exp_params)

    # Check if stage is finished

    finished = exp_data["results"]["test"]["finished"]

    if not finished:

        world_size = len(exp_params["settings"]["device_idxs"])
        manager = torch.multiprocessing.Manager()

        test_epoch_time_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        test_mean_loss_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        finished_mp = manager.Value("b", False)

        logger.print("Test - Multiprocessing start")

        torch.multiprocessing.spawn(
            execute_test_DDP,
            args=[
                world_size,
                command_args,
                exp_params,
                exp_data,
                test_epoch_time_list_mp,
                test_mean_loss_list_mp,
                finished_mp
            ],
            nprocs=world_size
        )
