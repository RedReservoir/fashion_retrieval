import os
import shutil
import sys
import pathlib
import pickle as pkl
import json

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

from time import time
from datetime import datetime

from itertools import chain
from functools import reduce

import json
import socket



########
# GLOBAL VARIABLES
########



datetime_now_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")



########
# COMPONENT FUNCTIONS
########



def create_backbone(backbone_class):

    if backbone_class == "ResNet50Backbone":
        backbone = backbones.ResNet50Backbone().to(first_device)
    if backbone_class == "EfficientNetB3Backbone":
        backbone = backbones.EfficientNetB3Backbone(batchnorm_track_runnning_stats=False).to(first_device)
    if backbone_class == "EfficientNetB4Backbone":
        backbone = backbones.EfficientNetB4Backbone(batchnorm_track_runnning_stats=False).to(first_device)
    if backbone_class == "EfficientNetB5Backbone":
        backbone = backbones.EfficientNetB5Backbone(batchnorm_track_runnning_stats=False).to(first_device)
    if backbone_class == "ConvNeXtTinyBackbone":
        backbone = backbones.ConvNeXtTinyBackbone().to(first_device)

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
        experiment_params
        ):

    # Load checkpoint

    experiment_checkpoint = torch.load(stage_1_experiment_checkpoint_filename)

    # Backbone

    backbone_class = experiment_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(first_device)

    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    # Optimizer

    optimizer_params = ret_head.parameters()

    optimizer_class = experiment_params["settings"]["stage_1"]["optimizer"]["class"]
    
    if optimizer_class == "Adam":
        optimizer = torch.optim.Adam(optimizer_params)
    if optimizer_class == "SGD":
        lr = experiment_params["settings"]["stage_1"]["optimizer"]["lr"]
        optimizer = torch.optim.SGD(optimizer_params, lr)

    optimizer.load_state_dict(experiment_checkpoint["optimizer_state_dict"])

    # Scheduler

    scheduler_class = experiment_params["settings"]["stage_1"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":
        gamma = experiment_params["settings"]["stage_1"]["scheduler"]["gamma"]
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
        experiment_params
        ):

    # Load checkpoint

    experiment_checkpoint = torch.load(stage_2_experiment_checkpoint_filename)

    # Backbone

    backbone_class = experiment_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(first_device)

    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    # Optimizer

    optimizer_params = chain(
        backbone.parameters(),
        ret_head.parameters()
    )

    optimizer_class = experiment_params["settings"]["stage_2"]["optimizer"]["class"]
    
    if optimizer_class == "Adam":
        optimizer = torch.optim.Adam(optimizer_params)
    if optimizer_class == "SGD":
        lr = experiment_params["settings"]["stage_2"]["optimizer"]["lr"]
        optimizer = torch.optim.SGD(optimizer_params, lr)

    optimizer.load_state_dict(experiment_checkpoint["optimizer_state_dict"])

    # Scheduler

    scheduler_class = experiment_params["settings"]["stage_2"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = experiment_params["settings"]["stage_1"]["scheduler"]["gamma"]
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


def initialize_stage_1_components(experiment_params):

    # Backbone

    backbone_class = experiment_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class)

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(first_device)
    
    # Optimizer

    optimizer_params = ret_head.parameters()

    optimizer_class = experiment_params["settings"]["stage_1"]["optimizer"]["class"]

    if optimizer_class == "Adam":

        lr = experiment_params["settings"]["stage_1"]["optimizer"]["lr"]

        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=lr
        )

    if optimizer_class == "SGD":

        lr = experiment_params["settings"]["stage_1"]["optimizer"]["lr"]
        momentum = experiment_params["settings"]["stage_1"]["optimizer"]["momentum"]

        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=lr,
            momentum=momentum
        )

    # Scheduler

    scheduler_class = experiment_params["settings"]["stage_1"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = experiment_params["settings"]["stage_1"]["scheduler"]["gamma"]

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

    # Early stopper

    patience = experiment_params["settings"]["stage_1"]["early_stopper"]["patience"]
    min_delta = experiment_params["settings"]["stage_1"]["early_stopper"]["min_delta"]

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
        experiment_params
        ):

    # Load checkpoint

    experiment_checkpoint = torch.load(best_stage_1_experiment_checkpoint_filename)

    # Backbone

    backbone_class = experiment_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])

    # Heads

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(first_device)

    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    # Optimizer

    optimizer_params = chain(
        backbone.parameters(),
        ret_head.parameters()
    )

    optimizer_class = experiment_params["settings"]["stage_2"]["optimizer"]["class"]

    if optimizer_class == "Adam":

        lr = experiment_params["settings"]["stage_2"]["optimizer"]["lr"]

        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=lr
        )

    if optimizer_class == "SGD":

        lr = experiment_params["settings"]["stage_2"]["optimizer"]["lr"]
        momentum = experiment_params["settings"]["stage_2"]["optimizer"]["momentum"]

        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=lr,
            momentum=momentum
        )

    # Scheduler

    scheduler_class = experiment_params["settings"]["stage_2"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = experiment_params["settings"]["stage_2"]["scheduler"]["gamma"]

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

    # Early stopper

    patience = experiment_params["settings"]["stage_2"]["early_stopper"]["patience"]
    min_delta = experiment_params["settings"]["stage_2"]["early_stopper"]["min_delta"]

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



def save_experiment_data(
        experiment_data_filename,
        experiment_data
        ):

    with open(experiment_data_filename, 'w') as experiment_data_file:
        json.dump(experiment_data, experiment_data_file, indent=2)


def load_experiment_data(
        experiment_data_filename
        ):

    with open(experiment_data_filename, 'r') as experiment_data_file:
        experiment_data = json.load(experiment_data_file)

    return experiment_data


def initialize_stage_1_experiment_data(
        experiment_data,
        experiment_params
        ):

    # Settings

    experiment_data["settings"]["stage_1"] = {}
    
    stage_description = "Train with frozen backbone"
    experiment_data["settings"]["stage_1"]["description"] = stage_description

    experiment_data["settings"]["stage_1"]["learning_rate_list"] = []
    
    ## Optimizer

    optimizer_class = experiment_params["settings"]["stage_1"]["optimizer"]["class"]

    if optimizer_class == "Adam":

        lr = experiment_params["settings"]["stage_1"]["optimizer"]["lr"]

        experiment_data["settings"]["stage_1"]["optimizer"] = {
            "class": "Adam",
            "lr": lr
        }

    if optimizer_class == "SGD":

        lr = experiment_params["settings"]["stage_1"]["optimizer"]["lr"]
        momentum = experiment_params["settings"]["stage_1"]["optimizer"]["momentum"]

        experiment_data["settings"]["stage_1"]["optimizer"] = {
            "class": "SGD",
            "lr": lr,
            "momentum": momentum
        }

    ## Scheduler
    
    scheduler_class = experiment_params["settings"]["stage_1"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = experiment_params["settings"]["stage_1"]["scheduler"]["gamma"]

        experiment_data["settings"]["stage_1"]["scheduler"] = {
            "class": "ExponentialLR",
            "gamma": gamma
        }

    ## Early stopper

    patience = experiment_params["settings"]["stage_1"]["early_stopper"]["patience"]
    min_delta = experiment_params["settings"]["stage_1"]["early_stopper"]["min_delta"]

    experiment_data["settings"]["stage_1"]["early_stopping"] = {
        "patience": patience,
        "min_delta": min_delta
    }

    # Other

    experiment_data["settings"]["stage_1"]["extra"] = [
        "automatic_mixed_precision"
    ]

    # Results

    experiment_data["results"]["stage_1"] = {}

    experiment_data["results"]["stage_1"]["train_epoch_time_list"] = []
    experiment_data["results"]["stage_1"]["val_epoch_time_list"] = []

    experiment_data["results"]["stage_1"]["mean_train_loss_list"] = []
    experiment_data["results"]["stage_1"]["mean_val_loss_list"] = []

    experiment_data["results"]["stage_1"]["num_epochs"] = 0
    experiment_data["results"]["stage_1"]["finished"] = False

    return experiment_data


def initialize_stage_2_experiment_data(experiment_data, experiment_params):

    # Settings

    experiment_data["settings"]["stage_2"] = {}

    stage_description = "Train entire model"
    experiment_data["settings"]["stage_2"]["description"] = stage_description
        
    experiment_data["settings"]["stage_2"]["learning_rate_list"] = []
    
    ## Optimizer

    optimizer_class = experiment_params["settings"]["stage_2"]["optimizer"]["class"]

    if optimizer_class == "Adam":

        lr = experiment_params["settings"]["stage_2"]["optimizer"]["lr"]

        experiment_data["settings"]["stage_2"]["optimizer"] = {
            "class": "Adam",
            "lr": lr
        }

    if optimizer_class == "SGD":

        lr = experiment_params["settings"]["stage_2"]["optimizer"]["lr"]
        momentum = experiment_params["settings"]["stage_2"]["optimizer"]["momentum"]

        experiment_data["settings"]["stage_2"]["optimizer"] = {
            "class": "SGD",
            "lr": lr,
            "momentum": momentum
        }

    ## Scheduler
    
    scheduler_class = experiment_params["settings"]["stage_2"]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = experiment_params["settings"]["stage_2"]["scheduler"]["gamma"]

        experiment_data["settings"]["stage_2"]["scheduler"] = {
            "class": "ExponentialLR",
            "gamma": gamma
        }

    ## Early stopper

    patience = experiment_params["settings"]["stage_2"]["early_stopper"]["patience"]
    min_delta = experiment_params["settings"]["stage_2"]["early_stopper"]["min_delta"]

    experiment_data["settings"]["stage_2"]["early_stopping"] = {
        "patience": patience,
        "min_delta": min_delta
    }

    # Other

    experiment_data["settings"]["stage_2"]["extra"] = [
        "automatic_mixed_precision"
    ]
    
    # Results

    experiment_data["results"]["stage_2"] = {}

    experiment_data["results"]["stage_2"]["train_epoch_time_list"] = []
    experiment_data["results"]["stage_2"]["val_epoch_time_list"] = []

    experiment_data["results"]["stage_2"]["mean_train_loss_list"] = []
    experiment_data["results"]["stage_2"]["mean_val_loss_list"] = []

    experiment_data["results"]["stage_2"]["num_epochs"] = 0
    experiment_data["results"]["stage_2"]["finished"] = False

    return experiment_data


def initialize_test_experiment_data(
        experiment_data,
        experiment_params
        ):

    # Settings

    experiment_data["settings"]["test"] = {}
    
    stage_description = "Test final model performance"
    experiment_data["settings"]["test"]["description"] = stage_description
    
    # Results

    experiment_data["results"]["test"] = {}

    experiment_data["results"]["test"]["finished"] = False

    return experiment_data



########
# TRAIN & EVAL FUNCTIONS
########



def train_epoch(data_loader, max_acc_iter=1, with_tqdm=True):

    backbone.train()
    ret_head.train()

    total_loss_item = 0
    curr_acc_iter = 0

    loader_gen = data_loader
    if with_tqdm: loader_gen = tqdm(loader_gen)

    for batch in loader_gen:

        batch_loss = batch_evaluation(batch)
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


def eval_epoch(data_loader, with_tqdm=True):

    backbone.eval()
    ret_head.eval()
    
    total_loss_item = 0

    with torch.no_grad():

        loader_gen = data_loader
        if with_tqdm: loader_gen = tqdm(loader_gen)

        for batch in loader_gen:

            batch_loss = batch_evaluation(batch)
            total_loss_item += batch_loss.sum().item()

    return total_loss_item


def batch_evaluation(batch):

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
# MAIN SCRIPT
########



if __name__ == "__main__":

    ########
    # COMMAND ARGS
    ########

    experiment_params_filename = os.path.join(pathlib.Path.home(), "fashion_retrieval", "exp_params", sys.argv[1])
    terminal_silent = False
    terminal_tqdm = True
    reset_experiment = False

    argidx = 2
    while argidx < len(sys.argv):

        if sys.argv[argidx] == "--silent":
            terminal_silent = True
            argidx += 1

        elif sys.argv[argidx] == "--notqdm":
            terminal_tqdm = False
            argidx += 1

        elif sys.argv[argidx] == "--reset":
            reset_experiment = True
            argidx += 1

    with_tqdm = terminal_tqdm and not terminal_silent

    ########
    # PREREQUISITES
    ########

    # Read experiment params

    experiment_params_file = open(experiment_params_filename, 'r')
    experiment_params = json.load(experiment_params_file)
    experiment_params_file.close()

    # Check and create experiment directory

    experiment_name = experiment_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)
    
    if os.path.exists(experiment_dirname) and reset_experiment:
        shutil.rmtree(experiment_dirname)

    if not os.path.exists(experiment_dirname):
        os.mkdir(experiment_dirname)

    # Prepare logger

    log_filename = "logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not terminal_silent: logger_streams.append(sys.stdout)

    logger = utils.log.Logger(logger_streams)

    # Load or create experiment data
    
    experiment_data_filename = os.path.join(
        experiment_dirname, "experiment_data.json"
    )

    if not os.path.exists(experiment_data_filename):

        experiment_data = {}
        experiment_data["experiment_name"] = experiment_name
        experiment_data["settings"] = {}
        experiment_data["results"] = {}

        experiment_data_filename = os.path.join(
            experiment_dirname, "experiment_data.json"
        )
        
        save_experiment_data(experiment_data_filename, experiment_data)
    
        logger.print("Starting experiment at {:s}".format(datetime_now_str))

    else:

        logger.print("Resuming experiment at {:s}".format(datetime_now_str))

    experiment_data = load_experiment_data(experiment_data_filename)

    ########
    # INITIALIZATION
    ########

    # Prepare GPUs

    logger.print("Selecting GPUs")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_idxs = experiment_params["settings"]["device_idxs"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in device_idxs])

    first_device = torch.device("cuda:0")
    device = torch.device("cuda")

    experiment_data["settings"]["gpu_usage"] = utils.mem.list_gpu_data(device_idxs)
    experiment_data["settings"]["hostname"] = socket.gethostname()

    torch.cuda.empty_cache()

    logger.print("Hostname: " + socket.gethostname())
    logger.print("Selected GPUs:")
    logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=2))

    # Create datasets and data loaders

    logger.print("Creating datasets")

    batch_size = experiment_params["settings"]["data_loading"]["batch_size"]
    num_workers = experiment_params["settings"]["data_loading"]["num_workers"]

    backbone_class = experiment_params["settings"]["backbone"]["class"]

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

    ctsrbm_dataset = deep_fashion.ConsToShopClothRetrBM(ctsrbm_dataset_dir, image_transform=backbone_image_transform, neg_aux_filename_id="test")

    train_idxs = ctsrbm_dataset.get_split_mask_idxs("train")
    test_idxs = ctsrbm_dataset.get_split_mask_idxs("test")
    val_idxs = ctsrbm_dataset.get_split_mask_idxs("val")

    cutdown_ratio = 1
    if cutdown_ratio != 1:

        logger.print("Cutting down datasets to {:5.2f}%".format(cutdown_ratio * 100))

        train_idxs = utils.list.cutdown_list(train_idxs, cutdown_ratio)
        test_idxs = utils.list.cutdown_list(test_idxs, cutdown_ratio)
        val_idxs = utils.list.cutdown_list(val_idxs, cutdown_ratio)

    if len(device_idxs) > 1:

        logger.print("Making dataset size multiple of batch size")

        train_idxs = train_idxs[:(len(train_idxs) // batch_size) * batch_size]
        test_idxs = test_idxs[:(len(test_idxs) // batch_size) * batch_size]
        val_idxs = val_idxs[:(len(val_idxs) // batch_size) * batch_size]

    ctsrbm_train_dataset = Subset(ctsrbm_dataset, train_idxs)
    ctsrbm_test_dataset = Subset(ctsrbm_dataset, test_idxs)
    ctsrbm_val_dataset = Subset(ctsrbm_dataset, val_idxs)

    logger.print("Final dataset sizes:")

    logger.print("  {:40s} | len: {:10d}".format("ctsrbm_train_dataset", len(ctsrbm_train_dataset)))
    logger.print("  {:40s} | len: {:10d}".format("ctsrbm_test_dataset", len(ctsrbm_test_dataset)))
    logger.print("  {:40s} | len: {:10d}".format("ctsrbm_val_dataset", len(ctsrbm_val_dataset)))

    logger.print("Creating data loaders")

    train_loader = DataLoader(ctsrbm_train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(ctsrbm_test_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(ctsrbm_val_dataset, batch_size=batch_size, num_workers=num_workers)

    experiment_data["settings"]["datasets"] = [
        "DeepFashion Consumer-to-shop Clothes Retrieval Benchmark"
    ]

    experiment_data["settings"]["data_loading"] = {
        "batch_size": batch_size,
        "num_workers": num_workers
    }

    # Model & head experiment data

    experiment_data["settings"]["backbone"] = {
        "class": experiment_params["settings"]["backbone"]["class"]
    }

    experiment_data["settings"]["heads"] = []
    experiment_data["settings"]["heads"].append({
        "class": "RetHead"
    })

    ########
    # STAGE 1
    ########

    # Initialize experiment data

    logger.print("Entering Stage 1")

    if utils.dict.chain_get(experiment_data, "settings", "stage_1") is None:

        experiment_data = initialize_stage_1_experiment_data(experiment_data, experiment_params)

    # Check if finished

    finished = experiment_data["results"]["stage_1"]["finished"]
    
    if finished:
    
        logger.print("Skipping Stage 1 - Finished")

    else:
    
        ## Initialize control variables

        max_epochs = experiment_params["settings"]["stage_1"]["max_epochs"]
        num_epochs = experiment_data["results"]["stage_1"]["num_epochs"]

        max_acc_iter = utils.dict.chain_get(
            experiment_params,
            "settings", "stage_1", "max_acc_iter",
            default=1
        )
        
        ## Initialize or load components

        logger.print("Preparing components")

        if num_epochs == 0:

            logger.print("  Initializing components")

            backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
            initialize_stage_1_components(
                experiment_params
            )

            logger.print("  Initialized components")

        else:

            logger.print("  Loading last epoch model from \"last_stage_1_ckp\"")

            last_stage_1_experiment_checkpoint_filename = os.path.join(
                experiment_dirname, "last_stage_1_ckp.pth"
            )

            backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
            load_stage_1_experiment_checkpoint(
                last_stage_1_experiment_checkpoint_filename,
                experiment_params
            )

            logger.print("  Loaded last epoch model from \"last_stage_1_ckp\"")

        ret_model = models.BackboneAndHead(backbone, ret_head).to(first_device)
        if len(device_idxs) > 1: ret_model = torch.nn.DataParallel(ret_model, device_ids=list(range(len(device_idxs))))

        scaler = torch.cuda.amp.GradScaler()

        for param in backbone.parameters():
            param.requires_grad = False

        ## Training loop

        logger.print("Entering training loop")

        while not finished:

            num_epochs += 1

            logger.print("Epoch {:d}/{:d}".format(num_epochs, max_epochs))

            ### Training & validation

            start_time = time()
            train_loss = train_epoch(train_loader, max_acc_iter=max_acc_iter, with_tqdm=with_tqdm) 
            end_time = time()

            train_epoch_time = end_time - start_time
            mean_train_loss = train_loss / len(ctsrbm_train_dataset)

            logger.print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(train_epoch_time)))
            logger.print("  Mean train loss: {:.2e}".format(mean_train_loss))

            start_time = time()
            val_loss = eval_epoch(val_loader, with_tqdm=with_tqdm) 
            end_time = time()

            val_epoch_time = end_time - start_time
            mean_val_loss = val_loss / len(ctsrbm_val_dataset)

            logger.print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(val_epoch_time)))
            logger.print("  Mean val loss:   {:.2e}".format(mean_val_loss))

            scheduler.step()

            logger.print("  Current memory usage:")
            logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))

            ### Track results

            experiment_data["results"]["stage_1"]["train_epoch_time_list"].append(train_epoch_time)
            experiment_data["results"]["stage_1"]["val_epoch_time_list"].append(val_epoch_time)

            experiment_data["results"]["stage_1"]["mean_train_loss_list"].append(mean_train_loss)
            experiment_data["results"]["stage_1"]["mean_val_loss_list"].append(mean_val_loss)

            experiment_data["settings"]["stage_1"]["learning_rate_list"].append(scheduler.get_last_lr()[0])

            ### Num epochs

            if num_epochs >= max_epochs:
                
                finished = True

                logger.print("Maximum number of epochs reached")

            ### Early stopping

            if early_stopper.early_stop(mean_val_loss):

                finished = True

                logger.print("Early stopping training")

            with utils.sig.DelayedInterrupt():

                ### Save best component checkpoint

                if best_tracker.is_best(mean_val_loss):

                    logger.print("  Saving currently best model to \"best_stage_1_ckp.pth\"")

                    best_stage_1_experiment_checkpoint_filename = os.path.join(
                        experiment_dirname, "best_stage_1_ckp.pth"
                    )

                    save_experiment_checkpoint(
                        best_stage_1_experiment_checkpoint_filename,
                        backbone,
                        ret_head,
                        optimizer,
                        scheduler,
                        early_stopper,
                        best_tracker
                        )
                    
                    logger.print("  Saved currently best model to \"best_stage_1_ckp.pth\"")

                ### Save experiment data

                logger.print("  Saving experiment data to \"experiment_data.pth\"")

                experiment_data["results"]["stage_1"]["num_epochs"] = num_epochs
                experiment_data["results"]["stage_1"]["finished"] = finished

                experiment_data_filename = os.path.join(
                    experiment_dirname, "experiment_data.json"
                )

                save_experiment_data(
                    experiment_data_filename, experiment_data
                )

                logger.print("  Saved experiment data to \"experiment_data.pth\"")

                ### Save last component checkpoint

                logger.print("  Saving last epoch model to \"last_stage_1_ckp.pth\"")
                
                last_stage_1_experiment_checkpoint_filename = os.path.join(
                    experiment_dirname, "last_stage_1_ckp.pth"
                )

                save_experiment_checkpoint(
                    last_stage_1_experiment_checkpoint_filename,
                    backbone,
                    ret_head,
                    optimizer,
                    scheduler,
                    early_stopper,
                    best_tracker
                )

                logger.print("  Saved last epoch model to \"last_stage_1_ckp.pth\"")

    ########
    # STAGE 2
    ########

    # Initialize experiment data

    logger.print("Entering Stage 2")

    if utils.dict.chain_get(experiment_data, "settings", "stage_2") is None:

        experiment_data = initialize_stage_2_experiment_data(experiment_data, experiment_params)

    # Check if finished

    finished = experiment_data["results"]["stage_2"]["finished"]
    
    if finished:
    
        logger.print("Skipping Stage 2 - Finished")

    else:
    
        ## Initialize control variables

        max_epochs = experiment_params["settings"]["stage_2"]["max_epochs"]
        num_epochs = experiment_data["results"]["stage_2"]["num_epochs"]

        max_acc_iter = utils.dict.chain_get(
            experiment_params,
            "settings", "stage_2", "max_acc_iter",
            default=1
        )

        ## Initialize or load components

        logger.print("Preparing components")

        if num_epochs == 0:

            logger.print("  Loading best model from previous stage from \"best_stage_1_ckp\"")

            best_stage_1_experiment_checkpoint_filename = os.path.join(
                experiment_dirname, "best_stage_1_ckp.pth"
            )

            backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
            initialize_stage_2_components(
                best_stage_1_experiment_checkpoint_filename,
                experiment_params
            )

            logger.print("  Loaded best model from previous stage from \"best_stage_1_ckp\"")

        else:

            logger.print("  Loading last epoch model from \"last_stage_2_ckp\"")

            last_stage_2_experiment_checkpoint_filename = os.path.join(
                experiment_dirname, "last_stage_2_ckp.pth"
            )

            backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
            load_stage_2_experiment_checkpoint(
                last_stage_2_experiment_checkpoint_filename,
                experiment_params
            )

            logger.print("  Loaded last epoch model from \"last_stage_2_ckp\"")

        ret_model = models.BackboneAndHead(backbone, ret_head).to(first_device)
        if len(device_idxs) > 1: ret_model = torch.nn.DataParallel(ret_model, device_ids=list(range(len(device_idxs))))

        scaler = torch.cuda.amp.GradScaler()

        for param in backbone.parameters():
            param.requires_grad = True

        ## Training loop

        logger.print("Entering training loop")

        while not finished:

            num_epochs += 1

            logger.print("Epoch {:d}/{:d}".format(num_epochs, max_epochs))

           ### Training & validation

            start_time = time()
            train_loss = train_epoch(train_loader, max_acc_iter=max_acc_iter, with_tqdm=with_tqdm) 
            end_time = time()

            train_epoch_time = end_time - start_time
            mean_train_loss = train_loss / len(ctsrbm_train_dataset)

            logger.print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(train_epoch_time)))
            logger.print("  Mean train loss: {:.2e}".format(mean_train_loss))

            start_time = time()
            val_loss = eval_epoch(val_loader, with_tqdm=with_tqdm) 
            end_time = time()

            val_epoch_time = end_time - start_time
            mean_val_loss = val_loss / len(ctsrbm_val_dataset)

            logger.print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(val_epoch_time)))
            logger.print("  Mean val loss:   {:.2e}".format(mean_val_loss))

            scheduler.step()

            logger.print("  Current memory usage:")
            logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))

            ### Track results

            experiment_data["results"]["stage_2"]["train_epoch_time_list"].append(train_epoch_time)
            experiment_data["results"]["stage_2"]["val_epoch_time_list"].append(val_epoch_time)

            experiment_data["results"]["stage_2"]["mean_train_loss_list"].append(mean_train_loss)
            experiment_data["results"]["stage_2"]["mean_val_loss_list"].append(mean_val_loss)

            experiment_data["settings"]["stage_2"]["learning_rate_list"].append(scheduler.get_last_lr()[0])

            ### Num epochs

            if num_epochs >= max_epochs:
                
                finished = True

                logger.print("Maximum number of epochs reached")

            ### Early stopping

            if early_stopper.early_stop(mean_val_loss):

                finished = True

                logger.print("Early stopping training")

            with utils.sig.DelayedInterrupt():

                ### Save best component checkpoint

                if best_tracker.is_best(mean_val_loss):

                    logger.print("  Saving currently best model to \"best_stage_2_ckp.pth\"")

                    best_stage_2_experiment_checkpoint_filename = os.path.join(
                        experiment_dirname, "best_stage_2_ckp.pth"
                    )

                    save_experiment_checkpoint(
                        best_stage_2_experiment_checkpoint_filename,
                        backbone,
                        ret_head,
                        optimizer,
                        scheduler,
                        early_stopper,
                        best_tracker
                        )

                    logger.print("  Saved currently best model to \"best_stage_2_ckp.pth\"")

                ### Save experiment data

                logger.print("  Saving experiment data to \"experiment_data.pth\"")

                experiment_data["results"]["stage_2"]["num_epochs"] = num_epochs
                experiment_data["results"]["stage_2"]["finished"] = finished

                experiment_data_filename = os.path.join(
                    experiment_dirname, "experiment_data.json"
                )

                save_experiment_data(
                    experiment_data_filename, experiment_data
                )

                logger.print("  Saved experiment data to \"experiment_data.pth\"")

                ### Save last component checkpoint

                logger.print("  Saving last epoch model to \"last_stage_2_ckp.pth\"")

                last_stage_2_experiment_checkpoint_filename = os.path.join(
                    experiment_dirname, "last_stage_2_ckp.pth"
                )

                save_experiment_checkpoint(
                    last_stage_2_experiment_checkpoint_filename,
                    backbone,
                    ret_head,
                    optimizer,
                    scheduler,
                    early_stopper,
                    best_tracker
                )

                logger.print("  Saved last epoch model to \"last_stage_2_ckp.pth\"")

    ########
    # TEST
    ########

    # Initialize experiment data

    logger.print("Entering Test phase")

    if utils.dict.chain_get(experiment_data, "settings", "test") is None:

        experiment_data = initialize_test_experiment_data(experiment_data, experiment_params)

    # Check if finished

    finished = experiment_data["results"]["test"]["finished"]
    
    if finished:
    
        logger.print("Skipping Test phase - Finished")

    else:

        ## Initialize or load components

        logger.print("Preparing components")

        logger.print("  Loading best model from previous stage from \"best_stage_2_ckp\"")

        best_stage_2_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "best_stage_2_ckp.pth"
        )

        backbone, ret_head, optimizer, scheduler, early_stopper, best_tracker =\
        load_stage_2_experiment_checkpoint(
            best_stage_2_experiment_checkpoint_filename,
            experiment_params
        )
            
        logger.print("  Loaded best model from previous stage from \"best_stage_2_ckp\"")

        ret_model = models.BackboneAndHead(backbone, ret_head).to(first_device)
        if len(device_idxs) > 1: ret_model = torch.nn.DataParallel(ret_model, device_ids=list(range(len(device_idxs))))

        scaler = torch.cuda.amp.GradScaler()

        ### Testing

        start_time = time()
        test_loss = eval_epoch(test_loader, with_tqdm=with_tqdm)
        end_time = time()

        test_epoch_time = end_time - start_time
        mean_test_loss = test_loss / len(ctsrbm_train_dataset)

        logger.print("  Test epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(test_epoch_time)))
        logger.print("  Mean test loss: {:.2e}".format(mean_test_loss))

        logger.print("  Current memory usage:")
        logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))
        
        ### Track results

        experiment_data["results"]["test"]["test_epoch_time"] = test_epoch_time
        experiment_data["results"]["test"]["mean_test_loss"] = mean_test_loss

        finished = True        

        with utils.sig.DelayedInterrupt():

            ### Save experiment data

            logger.print("  Saving experiment data to \"experiment_data.pth\"")

            experiment_data["results"]["test"]["finished"] = finished

            experiment_data_filename = os.path.join(
                experiment_dirname, "experiment_data.json"
            )

            save_experiment_data(
                experiment_data_filename, experiment_data
            )
                
            logger.print("  Saved experiment data to \"experiment_data.pth\"")
