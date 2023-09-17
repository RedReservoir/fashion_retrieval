import os
import shutil
import sys
import pathlib
import dill
import pickle as pkl
import json
import argparse
import random as rd

from contextlib import nullcontext

import numpy as np
import pandas as pd

import torch
import torchvision

from torch.utils.data import DataLoader, Subset

from fir.datasets import deep_fashion_ctsrbm
from fir.arch import backbones_cnn, backbones_trf, models, heads

import fir.utils.train
import fir.utils.log
import fir.utils.dict
import fir.utils.list
import fir.utils.mem
import fir.utils.sig
import fir.utils.time

from tqdm import tqdm

from time import time
from datetime import datetime

from itertools import chain
from functools import reduce

import json
import socket

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pprint



########
# COMPONENT FUNCTIONS
########



def save_experiment_checkpoint(
        experiment_checkpoint_filename,
        backbone,
        ret_head,
        optimizer,
        scheduler,
        early_stopper,
        best_tracker,
        grad_scaler
        ):

    experiment_checkpoint = {}

    experiment_checkpoint["backbone_state_dict"] = backbone.state_dict()
    experiment_checkpoint["ret_head_state_dict"] = ret_head.state_dict()
    experiment_checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    experiment_checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    experiment_checkpoint["early_stopper_state_dict"] = early_stopper.state_dict()
    experiment_checkpoint["best_tracker_state_dict"] = best_tracker.state_dict()
    if grad_scaler is not None:
        experiment_checkpoint["grad_scaler_state_dict"] = grad_scaler.state_dict()

    torch.save(experiment_checkpoint, experiment_checkpoint_filename)


def create_backbone(backbone_options):

    backbone_class = backbone_options["class"]
    backbone_img_size = backbone_options.get("img_size", None)

    # ResNet

    if backbone_class == "ResNet50Backbone":
        backbone = backbones_cnn.ResNet50Backbone()

    # EfficientNet
    
    if backbone_class == "EfficientNetB2Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = backbones_cnn.EfficientNetB2Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    if backbone_class == "EfficientNetB3Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = backbones_cnn.EfficientNetB3Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    if backbone_class == "EfficientNetB4Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = backbones_cnn.EfficientNetB4Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    if backbone_class == "EfficientNetB5Backbone":
        batchnorm_track_runnning_stats = backbone_options.get("batchnorm_track_runnning_stats", None)
        backbone = backbones_cnn.EfficientNetB5Backbone(
            batchnorm_track_runnning_stats=batchnorm_track_runnning_stats
        )

    if backbone_class == "EfficientNetV2SmallBackbone":
        backbone = backbones_cnn.EfficientNetV2SmallBackbone(
            backbone_img_size
        )
    
    # ConvNeXt

    if backbone_class == "ConvNeXtTinyBackbone":
        backbone = backbones_cnn.ConvNeXtTinyBackbone(contiguous_after_permute=True)
    
    # Cv Transformer
    
    if backbone_class == "CvTransformerB21I384D22kBackbone":
        backbone = backbones_trf.CvTransformerB21I384D22kBackbone(
            backbone_img_size
        )

    # Swin Transformer
    
    if backbone_class == "SwinTransformerV2TinyBackbone":
        backbone = backbones_trf.SwinTransformerV2TinyBackbone(
            backbone_img_size
        )

    return backbone


def create_ret_head(
        backbone,
        ret_head_options
        ):

    ret_head = heads.RetHead(backbone.feature_shape, ret_head_options["emb_size"])

    return ret_head


def create_optimizer(
        optimizer_params,
        optimizer_options,
        batch_size,
        num_devices
        ):
    
    optimizer_class = optimizer_options["class"]
    
    if optimizer_class == "Adam":
        lr = optimizer_options["lr"] * batch_size * num_devices
        optimizer = torch.optim.Adam(optimizer_params, lr)
    if optimizer_class == "SGD":
        lr = optimizer_options["lr"] * batch_size * num_devices
        momentum = optimizer_options["momentum"]
        optimizer = torch.optim.SGD(optimizer_params, lr, momentum)

    return optimizer


def create_scheduler(
        optimizer,
        scheduler_options
        ):

    scheduler_class = scheduler_options["class"]

    if scheduler_class == "ExponentialLR":
        gamma = scheduler_options["gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    return scheduler


def create_early_stopper(
        early_stopper_options
        ):

    patience = early_stopper_options["patience"]
    min_delta = early_stopper_options["min_delta"]

    early_stopper = fir.utils.train.EarlyStopper(patience, min_delta)

    return early_stopper


def create_best_tracker():

    early_stopper = fir.utils.train.BestTracker(0)

    return early_stopper


def create_grad_scaler():

    grad_scaler = torch.cuda.amp.GradScaler()

    return grad_scaler



########
# EXPERIMENT DATA FUNCTIONS
########



def save_json_dict(
        json_filename,
        json_dict
        ):

    with open(json_filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)


def load_json_dict(
        json_filename
        ):

    with open(json_filename, 'r') as json_file:
        json_dict = json.load(json_file)

    return json_dict


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
# DATASET AND LOADER FUNCTIONS
########



def compute_gradual_inc_cutdown_ratio(
        current_epoch_num,
        dgi_num_epochs,
        dgi_init_ratio
        ):

    cutdown_ratio = 1

    if current_epoch_num == 1:
        cutdown_ratio = dgi_init_ratio
    elif 2 <= current_epoch_num and current_epoch_num <= dgi_num_epochs - 1:
        cutdown_ratio = dgi_init_ratio + ((1 - dgi_init_ratio) / (2 ** (dgi_num_epochs - current_epoch_num)))

    return cutdown_ratio



########
# TRAIN & EVAL FUNCTIONS
########



def train_epoch(
        data_loader,
        ret_model,
        optimizer,
        grad_scaler,
        device,
        logger,
        grad_acc_iters=1,
        with_tqdm=True,
        rank=0
    ):

    ret_model.train()

    total_loss_item = 0
    curr_grad_acc_iter = 0

    with_amp = grad_scaler is not None

    loader_gen = data_loader
    if with_tqdm: loader_gen = tqdm(
            loader_gen,
            total=len(loader_gen),
            miniters=int(np.ceil(len(loader_gen)/20)),
            file=logger,
            ascii=False,
            desc="      Progress",
            ncols=0,
            mininterval=0,
            maxinterval=float("inf"),
            smoothing=0.99
            )
        
    for batch in loader_gen:
            
        batch_loss = batch_evaluation(batch, ret_model, device, logger, with_amp=with_amp)
        total_loss_item += batch_loss.sum().item()

        if grad_scaler is not None:
            grad_scaler.scale(batch_loss.mean() / grad_acc_iters).backward()
        else:
            (batch_loss.mean() / grad_acc_iters).backward()

        curr_grad_acc_iter += 1

        if curr_grad_acc_iter == grad_acc_iters:

            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            curr_grad_acc_iter = 0

    if curr_grad_acc_iter > 0:

        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

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
        if with_tqdm: loader_gen = tqdm(
                loader_gen,
                total=len(loader_gen),
                miniters=int(np.ceil(len(loader_gen)/20)),
                file=logger,
                ascii=False,
                desc="      Progress",
                ncols=0,
                mininterval=0,
                maxinterval=float("inf"),
                smoothing=0.99
                )

        for batch in loader_gen:

            batch_loss = batch_evaluation(batch, ret_model, device, logger, with_amp=False)
            total_loss_item += batch_loss.sum().item()

    return total_loss_item


def batch_evaluation(
        batch,
        ret_model,
        device,
        logger,
        with_amp
    ):

    anc_imgs = batch[0].to(device)
    pos_imgs = batch[1].to(device)
    neg_imgs = batch[2].to(device)

    with torch.cuda.amp.autocast() if with_amp else nullcontext():

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
    dataset_random_seed_mp,
    finished_mp
):


    # Prepare logger

    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "train_ret_DDP__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = fir.utils.log.Logger(logger_streams)

    sys.stderr = logger


    # PyTorch DDP stuff

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "{:d}".format(command_args.master_port)

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(rank)
    num_devices = len(exp_params["settings"]["device_idxs"])


    # Load or initialize components

    backbone = create_backbone(exp_params["settings"]["backbone"])
    backbone = backbone.to(device)

    ret_head = create_ret_head(
        backbone,
        exp_params["settings"]["ret_head"]
    )
    ret_head = ret_head.to(device)

    optimizer_params = ret_head.parameters()

    optimizer = create_optimizer(
        optimizer_params,
        exp_params["settings"]["stage_1"]["optimizer"],
        exp_params["settings"]["stage_1"]["data_loading"]["batch_size"],
        num_devices
        )

    scheduler = create_scheduler(
        optimizer,
        exp_params["settings"]["stage_1"]["scheduler"]
        )

    early_stopper = create_early_stopper(exp_params["settings"]["stage_1"]["early_stopper"])
    best_tracker = create_best_tracker()

    grad_scaler = None

    if exp_params["settings"]["stage_1"]["autocast"]["enabled"]:
        grad_scaler = torch.cuda.amp.GradScaler()

    num_epochs = exp_data["results"]["stage_1"]["num_epochs"]

    if num_epochs != 0:

        last_stage_1_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "train_ret_DDP__last_stage_1_ckp.pth", device
        )

        experiment_checkpoint = torch.load(last_stage_1_experiment_checkpoint_filename)

        backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])
        ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])
        optimizer.load_state_dict(experiment_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(experiment_checkpoint["scheduler_state_dict"])
        early_stopper.load_state_dict(experiment_checkpoint["early_stopper_state_dict"])
        best_tracker.load_state_dict(experiment_checkpoint["best_tracker_state_dict"])
        if exp_params["settings"]["stage_1"]["autocast"]["enabled"]:
            grad_scaler.load_state_dict(experiment_checkpoint["grad_scaler_state_dict"])

    for param in backbone.parameters():
        param.requires_grad = False  

    ret_model = models.BackboneAndHead(backbone, ret_head).to(device)
    ret_model = DDP(ret_model, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)


    # Dataset initialization

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")
    backbone_image_transform = backbone.get_image_transform()

    ctsrbm_dataset = deep_fashion_ctsrbm.ConsToShopClothRetrBmkDataset(ctsrbm_dataset_dir, img_transform=backbone_image_transform, neg_img_filename_list_id="test")

    ctsrbm_train_idxs = ctsrbm_dataset.get_subset_indices(split="train")
    ctsrbm_val_idxs = ctsrbm_dataset.get_subset_indices(split="val")

    ## Train idxs shuffle

    if rank == 0:

        train_random_seed = exp_params["settings"]["stage_1"]["data_loading"].get("train_random_seed", None)
        if train_random_seed is None: train_random_seed = rd.randrange(sys.maxsize)

        exp_data["settings"]["stage_1"]["data_loading"]["train_random_seed"] = train_random_seed
        dataset_random_seed_mp = train_random_seed      

    torch.distributed.barrier()

    state = np.random.get_state()
    np.random.seed = dataset_random_seed_mp
    np.random.shuffle(ctsrbm_train_idxs)
    np.random.set_state(state)

    ## General cutdown ratio

    cutdown_ratio = exp_params["settings"]["data_loading"].get("cutdown_ratio", None)
    if cutdown_ratio is not None:
        ctsrbm_train_idxs = fir.utils.list.cutdown_list(ctsrbm_train_idxs, cutdown_ratio)
        ctsrbm_val_idxs = fir.utils.list.cutdown_list(ctsrbm_val_idxs, cutdown_ratio)


    # Validation dataloader initialization

    batch_size = exp_params["settings"]["stage_1"]["data_loading"]["batch_size"]
    num_workers = exp_params["settings"]["stage_1"]["data_loading"]["num_workers"]

    ctsrbm_val_dataset = Subset(ctsrbm_dataset, ctsrbm_val_idxs)

    ctsrbm_val_loader = DataLoader(
        ctsrbm_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_val_dataset)
    )


    # Other settings

    with_tqdm = (not command_args.no_tqdm) and (rank == 0)

    grad_acc_iters = fir.utils.dict.chain_get(
        exp_params,
        "settings", "stage_1", "data_loading", "grad_acc_iters",
        default=1
    )    


    # Training loop

    if rank == 0:
        logger.print("  [Rank {:d}] Training Loop Begin".format(rank))

    num_epochs = exp_data["results"]["stage_1"]["num_epochs"]
    max_epochs = exp_params["settings"]["stage_1"]["max_epochs"]    

    if rank == 0:    
        finished_mp.value = exp_data["results"]["stage_1"]["finished"]

    torch.distributed.barrier()

    while not finished_mp.value:

        ## Epoch pre-processing

        num_epochs += 1
    
        if rank == 0:
            logger.print("  [Rank {:d}] Entering epoch {:d}".format(
                rank,
                num_epochs
            ))

        ## Train dataloader initialization

        inc_cutdown_ctsrbm_train_idxs = ctsrbm_train_idxs

        if exp_params["settings"]["stage_1"]["data_gradual_inc"]["enabled"]:

            inc_cutdown_ratio = compute_gradual_inc_cutdown_ratio(
                num_epochs,
                exp_params["settings"]["stage_1"]["data_gradual_inc"]["num_epochs"],
                exp_params["settings"]["stage_1"]["data_gradual_inc"]["init_perc"] / 100
                )

            if inc_cutdown_ratio != 1:

                inc_cutdown_ctsrbm_train_idxs = fir.utils.list.cutdown_list(inc_cutdown_ctsrbm_train_idxs, inc_cutdown_ratio)
                
                if rank == 0:
                    logger.print("  [Rank {:d}] Reduced size of train split to {:.2f}%".format(rank, inc_cutdown_ratio * 100))

        ctsrbm_train_dataset = Subset(ctsrbm_dataset, inc_cutdown_ctsrbm_train_idxs)

        ctsrbm_train_loader = DataLoader(
            ctsrbm_train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(ctsrbm_train_dataset)
        )

        ## Training

        if rank == 0:
            logger.print("    [Rank {:d}] Training...".format(rank))

        start_time = time()

        train_loss = train_epoch(
            ctsrbm_train_loader,
            ret_model,
            optimizer,
            grad_scaler,
            device,
            logger,
            grad_acc_iters=grad_acc_iters,
            with_tqdm=with_tqdm,
            rank=rank
        )

        end_time = time()

        train_epoch_time = end_time - start_time
        train_mean_loss = train_loss / len(ctsrbm_train_loader)

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Train epoch time: {:s}".format(
            rank,
            fir.utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Train mean loss:  {:.2e}".format(
            rank,
            train_mean_loss
        ))

        torch.distributed.barrier()

        ## Validation

        if rank == 0:
            logger.print("    [Rank {:d}] Validation...".format(rank))

        start_time = time()

        val_loss = eval_epoch(
            ctsrbm_val_loader,
            ret_model,
            device,
            logger,
            with_tqdm=with_tqdm
        )

        end_time = time()

        val_epoch_time = end_time - start_time
        val_mean_loss = val_loss / len(ctsrbm_val_loader)

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Val epoch time:   {:s}".format(
            rank,
            fir.utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Val mean loss:    {:.2e}".format(
            rank,
            train_mean_loss
        ))

        torch.distributed.barrier()

        ## Scheduler update

        if rank == 0:
            exp_data["settings"]["stage_1"]["learning_rate_list"].append(scheduler.get_last_lr()[0])

        scheduler.step()

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

        ## Memory profiling

        if rank == 0:
            logger.print("    [Rank {:d}] Current memory usage:".format(rank))
            logger.print(fir.utils.mem.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=6))
        
        ## Training conditions and checkpoints

        if rank == 0:

            ## Number of epochs

            if num_epochs >= max_epochs:
                finished_mp.value = True

            ## Early stopping

            if early_stopper.early_stop(sum(train_mean_loss_list_mp)):
                finished_mp.value = True

            ## Checkpoint saving

            with fir.utils.sig.DelayedInterrupt():

                ## Save best component checkpoint

                if best_tracker.is_best(sum(train_mean_loss_list_mp)):

                    best_stage_1_experiment_checkpoint_filename = os.path.join(
                        experiment_dirname, "train_ret_DDP__best_stage_1_ckp.pth"
                    )

                    logger.print("    [Rank {:d}] Saving currently best model to \"{:s}\"".format(
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
                        best_tracker,
                        grad_scaler
                        )

                    logger.print("    [Rank {:d}] Saved currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_1_experiment_checkpoint_filename
                    ))
                    
                ## Save experiment data

                exp_data["results"]["stage_1"]["num_epochs"] = num_epochs
                exp_data["results"]["stage_1"]["finished"] = finished_mp.value

                exp_data_filename = os.path.join(
                    experiment_dirname, "train_ret_DDP__data.json"
                )

                logger.print("    [Rank {:d}] Saving experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                save_json_dict(
                    exp_data_filename, exp_data
                )

                logger.print("    [Rank {:d}] Saved experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                ## Save last component checkpoint
                
                last_stage_1_experiment_checkpoint_filename = os.path.join(
                    experiment_dirname, "train_ret_DDP__last_stage_1_ckp.pth"
                )

                logger.print("    [Rank {:d}] Saving last epoch model to \"{:s}\"".format(
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
                    best_tracker,
                    grad_scaler
                )

                logger.print("    [Rank {:d}] Saved last epoch model to \"{:s}\"".format(
                    rank,
                    last_stage_1_experiment_checkpoint_filename
                ))

        torch.distributed.barrier()

    if rank == 0:
        logger.print("    [Rank {:d}] Training Loop End".format(rank))


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
    dataset_random_seed_mp,
    finished_mp
):


    # Prepare logger

    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "train_ret_DDP__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = fir.utils.log.Logger(logger_streams)

    sys.stderr = logger


    # PyTorch DDP stuff

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "{:d}".format(command_args.master_port)

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(rank)
    num_devices = len(exp_params["settings"]["device_idxs"])
    

    # Load or initialize components

    backbone = create_backbone(exp_params["settings"]["backbone"])
    backbone = backbone.to(device)

    ret_head = create_ret_head(
        backbone,
        exp_params["settings"]["ret_head"]
    )
    ret_head = ret_head.to(device)

    optimizer_params = chain(
        backbone.parameters(),
        ret_head.parameters()
    )

    optimizer = create_optimizer(
        optimizer_params,
        exp_params["settings"]["stage_2"]["optimizer"],
        exp_params["settings"]["stage_2"]["data_loading"]["batch_size"],
        num_devices
        )

    scheduler = create_scheduler(
        optimizer,
        exp_params["settings"]["stage_2"]["scheduler"]
        )

    early_stopper = create_early_stopper(exp_params["settings"]["stage_2"]["early_stopper"])
    best_tracker = create_best_tracker()

    grad_scaler = None

    if exp_params["settings"]["stage_2"]["autocast"]["enabled"]:
        grad_scaler = torch.cuda.amp.GradScaler()

    num_epochs = exp_data["results"]["stage_2"]["num_epochs"]

    if num_epochs == 0:

        best_stage_1_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "train_ret_DDP__best_stage_1_ckp.pth"
        )

        experiment_checkpoint = torch.load(best_stage_1_experiment_checkpoint_filename)

        backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])
        ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    else:

        last_stage_2_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "train_ret_DDP__last_stage_2_ckp.pth"
        )

        experiment_checkpoint = torch.load(last_stage_2_experiment_checkpoint_filename)

        backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])
        ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])
        optimizer.load_state_dict(experiment_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(experiment_checkpoint["scheduler_state_dict"])
        early_stopper.load_state_dict(experiment_checkpoint["early_stopper_state_dict"])
        best_tracker.load_state_dict(experiment_checkpoint["best_tracker_state_dict"])
        if exp_params["settings"]["stage_1"]["autocast"]["enabled"]:
            grad_scaler.load_state_dict(experiment_checkpoint["grad_scaler_state_dict"])

    ret_model = models.BackboneAndHead(backbone, ret_head).to(device)
    ret_model = DDP(ret_model, device_ids=[rank], broadcast_buffers=False)


    # Dataset initialization

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")
    backbone_image_transform = backbone.get_image_transform()

    ctsrbm_dataset = deep_fashion_ctsrbm.ConsToShopClothRetrBmkDataset(ctsrbm_dataset_dir, img_transform=backbone_image_transform, neg_img_filename_list_id="test")

    ctsrbm_train_idxs = ctsrbm_dataset.get_subset_indices(split="train")
    ctsrbm_val_idxs = ctsrbm_dataset.get_subset_indices(split="val")

    ## Train idxs shuffle

    if rank == 0:

        train_random_seed = exp_params["settings"]["stage_2"]["data_loading"].get("train_random_seed", None)
        if train_random_seed is None: train_random_seed = rd.randrange(sys.maxsize)

        exp_data["settings"]["stage_2"]["data_loading"]["train_random_seed"] = train_random_seed
        dataset_random_seed_mp = train_random_seed  

    torch.distributed.barrier()

    state = np.random.get_state()
    np.random.seed = dataset_random_seed_mp
    np.random.shuffle(ctsrbm_train_idxs)
    np.random.set_state(state)

    ## General cutdown ratio

    cutdown_ratio = exp_params["settings"]["data_loading"].get("cutdown_ratio", None)
    if cutdown_ratio is not None:
        ctsrbm_train_idxs = fir.utils.list.cutdown_list(ctsrbm_train_idxs, cutdown_ratio)
        ctsrbm_val_idxs = fir.utils.list.cutdown_list(ctsrbm_val_idxs, cutdown_ratio)


    # Validation dataloader initialization

    batch_size = exp_params["settings"]["stage_2"]["data_loading"]["batch_size"]
    num_workers = exp_params["settings"]["stage_2"]["data_loading"]["num_workers"]

    ctsrbm_val_dataset = Subset(ctsrbm_dataset, ctsrbm_val_idxs)

    ctsrbm_val_loader = DataLoader(
        ctsrbm_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_val_dataset)
    )


    # Other settings

    with_tqdm = (not command_args.no_tqdm) and (rank == 0)

    grad_acc_iters = fir.utils.dict.chain_get(
        exp_params,
        "settings", "stage_2", "data_loading", "grad_acc_iters",
        default=1
    )  


    # Training loop

    if rank == 0:
        logger.print("  [Rank {:d}] Training Loop Begin".format(rank))

    num_epochs = exp_data["results"]["stage_2"]["num_epochs"]
    max_epochs = exp_params["settings"]["stage_2"]["max_epochs"]   

    if rank == 0:    
        finished_mp.value = exp_data["results"]["stage_2"]["finished"]

    torch.distributed.barrier()

    while not finished_mp.value:

        ## Epoch pre-processing

        num_epochs += 1

        if rank == 0:
            logger.print("  [Rank {:d}] Entering epoch {:d}".format(
                rank,
                num_epochs
            ))

        ## Train dataloader initialization

        inc_cutdown_ctsrbm_train_idxs = ctsrbm_train_idxs

        if exp_params["settings"]["stage_2"]["data_gradual_inc"]["enabled"]:

            inc_cutdown_ratio = compute_gradual_inc_cutdown_ratio(
                num_epochs,
                exp_params["settings"]["stage_2"]["data_gradual_inc"]["num_epochs"],
                exp_params["settings"]["stage_2"]["data_gradual_inc"]["init_perc"] / 100
                )

            if inc_cutdown_ratio != 1:

                inc_cutdown_ctsrbm_train_idxs = fir.utils.list.cutdown_list(inc_cutdown_ctsrbm_train_idxs, inc_cutdown_ratio)
                
                if rank == 0:
                    logger.print("  [Rank {:d}] Reduced size of train split to {:.2f}%".format(rank, inc_cutdown_ratio * 100))

        ctsrbm_train_dataset = Subset(ctsrbm_dataset, inc_cutdown_ctsrbm_train_idxs)

        ctsrbm_train_loader = DataLoader(
            ctsrbm_train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(ctsrbm_train_dataset)
        )

        ## Training

        if rank == 0:
            logger.print("    [Rank {:d}] Training...".format(rank))

        start_time = time()

        train_loss = train_epoch(
            ctsrbm_train_loader,
            ret_model,
            optimizer,
            grad_scaler,
            device,
            logger,
            grad_acc_iters=grad_acc_iters,
            with_tqdm=with_tqdm
        )

        end_time = time()

        train_epoch_time = end_time - start_time
        train_mean_loss = train_loss / len(ctsrbm_train_loader)

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Train epoch time: {:s}".format(
            rank,
            fir.utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Train mean loss:  {:.2e}".format(
            rank,
            train_mean_loss
        ))

        torch.distributed.barrier()

        ## Validation

        if rank == 0:
            logger.print("    [Rank {:d}] Validation...".format(rank))

        start_time = time()

        val_loss = eval_epoch(
            ctsrbm_val_loader,
            ret_model,
            device,
            logger,
            with_tqdm=with_tqdm
        )

        end_time = time()

        val_epoch_time = end_time - start_time
        val_mean_loss = val_loss / len(ctsrbm_val_loader)

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Val epoch time:   {:s}".format(
            rank,
            fir.utils.time.sprint_fancy_time_diff(train_epoch_time)
        ))

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Val mean loss:    {:.2e}".format(
            rank,
            train_mean_loss
        ))

        torch.distributed.barrier()

        ## Scheduler update

        if rank == 0:
            exp_data["settings"]["stage_2"]["learning_rate_list"].append(scheduler.get_last_lr()[0])

        scheduler.step()

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

        ## Memory profiling

        if rank == 0:
            logger.print("    [Rank {:d}] Current memory usage:".format(rank))
            logger.print(fir.utils.mem.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=6))
        
        ## Training conditions and checkpoints

        if rank == 0:

            ## Number of epochs

            if num_epochs >= max_epochs:
                finished_mp.value = True

            ## Early stopping

            if early_stopper.early_stop(sum(train_mean_loss_list_mp)):
                finished_mp.value = True

            ## Checkpoint saving

            with fir.utils.sig.DelayedInterrupt():

                ## Save best component checkpoint

                if best_tracker.is_best(sum(train_mean_loss_list_mp)):

                    best_stage_2_experiment_checkpoint_filename = os.path.join(
                        experiment_dirname, "train_ret_DDP__best_stage_2_ckp.pth"
                    )

                    logger.print("    [Rank {:d}] Saving currently best model to \"{:s}\"".format(
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
                        best_tracker,
                        grad_scaler
                        )

                    logger.print("    [Rank {:d}] Saved currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_2_experiment_checkpoint_filename
                    ))
                    
                ## Save experiment data

                exp_data["results"]["stage_2"]["num_epochs"] = num_epochs
                exp_data["results"]["stage_2"]["finished"] = finished_mp.value

                exp_data_filename = os.path.join(
                    experiment_dirname, "train_ret_DDP__data.json"
                )

                logger.print("    [Rank {:d}] Saving experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                save_json_dict(
                    exp_data_filename, exp_data
                )

                logger.print("    [Rank {:d}] Saved experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                ## Save last component checkpoint
                
                last_stage_2_experiment_checkpoint_filename = os.path.join(
                    experiment_dirname, "train_ret_DDP__last_stage_2_ckp.pth"
                )

                logger.print("    [Rank {:d}] Saving last epoch model to \"{:s}\"".format(
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
                    best_tracker,
                    grad_scaler
                )

                logger.print("    [Rank {:d}] Saved last epoch model to \"{:s}\"".format(
                    rank,
                    last_stage_2_experiment_checkpoint_filename
                ))

        torch.distributed.barrier()

    if rank == 0:
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

    log_filename = "train_ret_DDP__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = fir.utils.log.Logger(logger_streams)

    sys.stderr = logger


    # PyTorch DDP stuff

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "{:d}".format(command_args.master_port)

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(rank)
    

    # Load or initialize components

    backbone = create_backbone(exp_params["settings"]["backbone"])
    backbone = backbone.to(device)

    ret_head = create_ret_head(
        backbone,
        exp_params["settings"]["ret_head"]
    )
    ret_head = ret_head.to(device)

    last_stage_2_experiment_checkpoint_filename = os.path.join(
        experiment_dirname, "train_ret_DDP__last_stage_2_ckp.pth"
    )

    experiment_checkpoint = torch.load(last_stage_2_experiment_checkpoint_filename)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])
    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])
        
    ret_model = models.BackboneAndHead(backbone, ret_head).to(device)
    ret_model = DDP(ret_model, device_ids=[rank], broadcast_buffers=False)


    # Dataset initialization

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")
    backbone_image_transform = backbone.get_image_transform()

    ctsrbm_dataset = deep_fashion_ctsrbm.ConsToShopClothRetrBmkDataset(ctsrbm_dataset_dir, img_transform=backbone_image_transform, neg_img_filename_list_id="test")

    ctsrbm_test_idxs = ctsrbm_dataset.get_subset_indices(split="test")

    ## General cutdown ratio

    cutdown_ratio = exp_params["settings"]["data_loading"].get("cutdown_ratio", None)
    if cutdown_ratio is not None:
        ctsrbm_test_idxs = fir.utils.list.cutdown_list(ctsrbm_test_idxs, cutdown_ratio)
    

    # Test dataloader initialization

    batch_size = exp_params["settings"]["test"]["data_loading"]["batch_size"]
    num_workers = exp_params["settings"]["test"]["data_loading"]["num_workers"]

    ctsrbm_test_dataset = Subset(ctsrbm_dataset, ctsrbm_test_idxs)

    ctsrbm_test_loader = DataLoader(
        ctsrbm_test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(ctsrbm_test_dataset)
    )


    # Other settings

    with_tqdm = (not command_args.no_tqdm) and (rank == 0)


    # Testing

    if rank == 0:
        logger.print("  [Rank {:d}] Testing Begin".format(rank))

    if rank == 0:    
        finished_mp.value = exp_data["results"]["test"]["finished"]

    torch.distributed.barrier()

    if not finished_mp.value:

        ## Test epoch

        if rank == 0:
            logger.print("    [Rank {:d}] Testing...".format(rank))

        start_time = time()

        test_loss = eval_epoch(
            ctsrbm_test_loader,
            ret_model,
            device,
            logger,
            with_tqdm=with_tqdm
        )

        end_time = time()

        test_epoch_time = end_time - start_time
        test_mean_loss = test_loss / len(ctsrbm_test_dataset)

        torch.distributed.barrier()

        logger.print("  [Rank {:d}] Test epoch time:  {:s}".format(
            rank,
            fir.utils.time.sprint_fancy_time_diff(test_epoch_time)
        ))

        torch.distributed.barrier()

        logger.print("  [Rank {:d}] Test mean loss:   {:.2e}".format(
            rank,
            test_mean_loss
        ))

        torch.distributed.barrier()

        ## Track results

        if rank == 0:
            logger.print("  [Rank {:d}] Current memory usage:".format(rank))
            logger.print(fir.utils.mem.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=4))
        
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

            with fir.utils.sig.DelayedInterrupt():

                ## Save experiment data

                exp_data["results"]["test"]["finished"] = finished_mp.value

                exp_data_filename = os.path.join(
                    experiment_dirname, "train_ret_DDP__data.json"
                )

                logger.print("  [Rank {:d}] Saving experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                save_json_dict(
                    exp_data_filename, exp_data
                )

                logger.print("  [Rank {:d}] Saved experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

    if rank == 0:
        logger.print("  [Rank {:d}] Testing End".format(rank))


    # PyTorch DDP stuff

    destroy_process_group()



########
# MAIN SCRIPT
########



if __name__ == "__main__":


    datetime_now_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")


    ####
    # COMMAND ARGUMENTS
    ####


    parser = argparse.ArgumentParser()
    parser.add_argument("exp_params_filename", help="filename of the experiment params json file")
    parser.add_argument("--master_port", type=int, default=12355, help="master port for multi-GPU communication")
    parser.add_argument("--terminal_silent", help="no terminal prints will be made", action="store_true")
    parser.add_argument("--no_tqdm", help="no tqdm bars will be shown", action="store_true")
    parser.add_argument("--reset_experiment", help="experiment directory will be reset", action="store_true")
    parser.add_argument("--autograd_anomaly", help="wether to use torch.autograd.set_detect_anomaly(True) for debugging", action="store_true")
    command_args = parser.parse_args()

    exp_params_filename = os.path.join(pathlib.Path.home(), "fashion_retrieval", command_args.exp_params_filename)

    if command_args.autograd_anomaly:
        torch.autograd.set_detect_anomaly(True)

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
    
    if os.path.exists(experiment_dirname) and command_args.reset_experiment:
        shutil.rmtree(experiment_dirname)

    if not os.path.exists(experiment_dirname):
        os.mkdir(experiment_dirname)
    

    ####
    # PREPARE LOGGER
    ####


    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "train_ret_DDP__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = fir.utils.log.Logger(logger_streams)

    sys.stderr = logger
    
    logger.print("Bash command:", " ".join(sys.argv))


    ####
    # PREPARE EXPERIMENT DATA
    ####

    
    exp_data_filename = os.path.join(
        experiment_dirname, "train_ret_DDP__data.json"
    )

    if not os.path.exists(exp_data_filename):

        exp_data = {}
        exp_data["script_name"] = sys.argv[0]
        exp_data["experiment_name"] = experiment_name
        exp_data["settings"] = {}
        exp_data["results"] = {}

        exp_data["settings"]["datasets"] = [
            "DeepFashion Consumer-to-shop Clothes Retrieval Benchmark"
        ]

        exp_data["settings"]["backbone"] = exp_params["settings"]["backbone"]

        exp_data["settings"]["heads"] = []
        exp_data["settings"]["heads"].append({
            "class": "RetHead"
        })

        exp_data["settings"]["command_args"] = vars(command_args)

        exp_data_filename = os.path.join(
            experiment_dirname, "train_ret_DDP__data.json"
        )
        
        save_json_dict(exp_data_filename, exp_data)

        logger.print("Starting run at {:s}".format(datetime_now_str))

    else:

        logger.print("Resuming run at {:s}".format(datetime_now_str))
    
    exp_data = load_json_dict(exp_data_filename)


    ####
    # GPU INITIALIZATION
    ####


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_idxs = exp_params["settings"]["device_idxs"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in device_idxs])

    torch.cuda.empty_cache()

    exp_data["settings"]["gpu_usage"] = fir.utils.mem.list_gpu_data(device_idxs)
    exp_data["settings"]["hostname"] = socket.gethostname()
    exp_data["settings"]["master_port"] = command_args.master_port


    ####
    # STAGE 1 - FROZEN BACKBONE
    ####


    # Initialize experiment data

    if fir.utils.dict.chain_get(exp_data, "settings", "stage_1") is None:
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
        dataset_random_seed_mp = manager.Value("f", 0)
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
                dataset_random_seed_mp,
                finished_mp
            ],
            nprocs=world_size
        )

    # Reload experiment data

    exp_data = load_json_dict(exp_data_filename)


    ####
    # STAGE 2 - UNFROZEN BACKBONE
    ####


    # Initialize experiment data

    if fir.utils.dict.chain_get(exp_data, "settings", "stage_2") is None:
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
        dataset_random_seed_mp = manager.Value("f", 0)
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
                dataset_random_seed_mp,
                finished_mp
            ],
            nprocs=world_size
        )

    # Reload experiment data

    exp_data = load_json_dict(exp_data_filename)


    ####
    # TEST
    ####


    # Initialize experiment data

    if fir.utils.dict.chain_get(exp_data, "settings", "test") is None:
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
