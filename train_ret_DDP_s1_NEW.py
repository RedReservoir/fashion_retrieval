import os
import shutil
import sys
import pathlib
import json
import argparse
import random as rd
import math

from tqdm import tqdm

from time import time
from datetime import datetime

from itertools import chain

import json
import socket

from contextlib import nullcontext

import numpy as np

########

import torch

from torch.utils.data import DataLoader, Subset

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

########

from src.datasets import deep_fashion_ctsrbm

import src.utils.train
import src.utils.log
import src.utils.dict
import src.utils.list
import src.utils.memory
import src.utils.signal
import src.utils.time
import src.utils.comps
import src.utils.json
import src.utils.dgi
import src.utils.nvgpu



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
            miniters=int(np.ceil(len(loader_gen)/40)),
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
                miniters=int(np.ceil(len(loader_gen)/40)),
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
    dataset_random_mode_mp,
    finished_mp
):


    # Prepare logger

    experiment_name = exp_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "train_ret_DDP_stage_1__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = src.utils.log.Logger(logger_streams)

    sys.stderr = logger


    # PyTorch DDP stuff

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "{:d}".format(command_args.master_port)

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(rank)


    # Load or initialize components

    backbone = src.utils.comps.create_backbone(exp_params["settings"]["backbone"])
    backbone = backbone.to(device)

    ret_head = src.utils.comps.create_head(
        backbone,
        exp_params["settings"]["head"]
    )
    ret_head = ret_head.to(device)

    optimizer_params = ret_head.parameters()

    batch_size = exp_params["settings"]["stage_1"]["data_loading"]["batch_size"]
    num_devices = len(exp_params["settings"]["device_idxs"])
    grad_acc_iters = src.utils.dict.chain_get(
        exp_params,
        "settings", "stage_1", "data_loading", "grad_acc_iters",
        default=1
    )    

    optimizer = src.utils.comps.create_optimizer(
        optimizer_params,
        exp_params["settings"]["stage_1"]["optimizer"],
        batch_size,
        num_devices,
        grad_acc_iters
        )

    scheduler = src.utils.comps.create_scheduler(
        optimizer,
        exp_params["settings"]["stage_1"]["scheduler"]
        )

    early_stopper = src.utils.comps.create_early_stopper(exp_params["settings"]["stage_1"]["early_stopper"])
    best_tracker = src.utils.comps.create_best_tracker()

    grad_scaler = None

    if exp_params["settings"]["stage_1"]["autocast"]["enabled"]:
        grad_scaler = torch.cuda.amp.GradScaler()

    num_epochs = exp_data["results"]["stage_1"]["num_epochs"]

    if num_epochs != 0:

        last_stage_1_experiment_checkpoint_filename = os.path.join(
            experiment_dirname, "train_ret_DDP_stage_1__last_ckp.pth"
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

    ret_model = torch.nn.Sequential(backbone, ret_head).to(device)
    ret_model = DDP(ret_model, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)


    # Dataset initialization

    ctsrbm_dataset_options = exp_data["settings"]["stage_1"]["ctsrbm_dataset"]

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")
    backbone_image_transform = backbone.get_image_transform()

    ctsrbm_dataset = deep_fashion_ctsrbm.ConsToShopClothRetrBmkDataset(
        ctsrbm_dataset_dir,
        img_transform=backbone_image_transform,
        neg_img_filename_list_id=ctsrbm_dataset_options["neg_img_filename_list_id"]
    )

    ctsrbm_train_idxs = ctsrbm_dataset.get_subset_indices(split="train")
    ctsrbm_val_idxs = ctsrbm_dataset.get_subset_indices(split="val")

    ## Train idxs shuffle

    if rank == 0:

        train_random_seed = ctsrbm_dataset_options.get("train_random_seed", "none")
        
        if type(train_random_seed) is str:
            dataset_random_mode_mp.value = train_random_seed
        if type(train_random_seed) is int:
            dataset_random_mode_mp.value = "fixed"
            dataset_random_seed_mp.value = train_random_seed

        exp_data["settings"]["stage_1"]["data_loading"]["train_random_seed"] = train_random_seed

    torch.distributed.barrier()

    if dataset_random_mode_mp.value == "random":
        np.random.shuffle(ctsrbm_train_idxs)
    if dataset_random_mode_mp.value == "fixed":
        state = np.random.get_state()
        np.random.seed(dataset_random_seed_mp.value)
        np.random.shuffle(ctsrbm_train_idxs)
        np.random.set_state(state)

    ## General cutdown ratio

    cutdown_ratio = ctsrbm_dataset_options.get("cutdown_ratio", 1)
    if cutdown_ratio != 1:
        ctsrbm_train_idxs = src.utils.list.cutdown_list(ctsrbm_train_idxs, cutdown_ratio)
        ctsrbm_val_idxs = src.utils.list.cutdown_list(ctsrbm_val_idxs, cutdown_ratio)


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

    with_tqdm = (not command_args.disable_tqdm) and (rank == 0)

    grad_acc_iters = src.utils.dict.chain_get(
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
        finished_mp.value = exp_data["results"]["stage_1"]["finished"] and not command_args.resume_experiment

    torch.distributed.barrier()

    while not finished_mp.value:


        ## Epoch pre-processing

        num_epochs += 1
    
        if rank == 0:
            logger.print("  [Rank {:d}] Entering epoch {:d}".format(
                rank,
                num_epochs
            ))


        ## Epoch 0 Evaluation

        if num_epochs == 1 and exp_params["settings"]["stage_1"].get("eval_epoch_0", False):

            ### Train dataloader initialization
            
            ctsrbm_train_dataset = Subset(ctsrbm_dataset, ctsrbm_train_idxs)

            ctsrbm_train_loader = DataLoader(
                ctsrbm_train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
                sampler=DistributedSampler(ctsrbm_train_dataset)
            )

            ### Training

            if rank == 0:
                logger.print("    [Rank {:d}] Epoch 0 Training...".format(rank))

            train_loss = eval_epoch(
                ctsrbm_train_loader,
                ret_model,
                device,
                logger,
                with_tqdm=with_tqdm
            )

            train_mean_loss = train_loss / len(ctsrbm_train_dataset)

            torch.distributed.barrier()

            logger.print("    [Rank {:d}] Epoch 0 Train mean loss:  {:.2e}".format(
                rank,
                train_mean_loss
            ))

            torch.distributed.barrier()

            ### Validation

            if rank == 0:
                logger.print("    [Rank {:d}] Epoch 0 Validation...".format(rank))

            val_loss = eval_epoch(
                ctsrbm_val_loader,
                ret_model,
                device,
                logger,
                with_tqdm=with_tqdm
            )

            val_mean_loss = val_loss / len(ctsrbm_val_dataset)

            torch.distributed.barrier()

            logger.print("    [Rank {:d}] Epoch 0 Val mean loss:    {:.2e}".format(
                rank,
                val_mean_loss
            ))

            torch.distributed.barrier()

            ### Track results

            train_mean_loss_list_mp[rank] = train_mean_loss
            val_mean_loss_list_mp[rank] = val_mean_loss            

            torch.distributed.barrier()

            if rank == 0:

                exp_data["results"]["stage_1"]["epoch_0_train_mean_loss_list"] = list(train_mean_loss_list_mp)
                exp_data["results"]["stage_1"]["epoch_0_val_mean_loss_list"] = list(val_mean_loss_list_mp)

            ### Memory profiling

            if rank == 0:

                logger.print("    [Rank {:d}] Current memory usage:".format(rank))
                logger.print(src.utils.nvgpu.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=6))


        ## Train dataloader initialization
        
        ctsrbm_train_dataset = Subset(ctsrbm_dataset, ctsrbm_train_idxs)

        if exp_params["settings"]["stage_1"]["data_gradual_inc"]["enabled"]:

            dgi_ratio = src.utils.dgi.compute_dgi_ratio(
                num_epochs,
                exp_params["settings"]["stage_1"]["data_gradual_inc"]["num_epochs"],
                exp_params["settings"]["stage_1"]["data_gradual_inc"]["init_perc"] / 100
                )

            if dgi_ratio != 1:

                ctsrbm_train_dataset = Subset(ctsrbm_dataset, src.utils.list.cutdown_list(ctsrbm_train_idxs, dgi_ratio))

                if rank == 0:
                    logger.print("  [Rank {:d}] Reduced size of train split to {:2.2f}%".format(rank, dgi_ratio * 100))

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
        train_mean_loss = train_loss / len(ctsrbm_train_dataset)

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Train epoch time: {:s}".format(
            rank,
            src.utils.time.sprint_fancy_time_diff(train_epoch_time)
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
        val_mean_loss = val_loss / len(ctsrbm_val_dataset)

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Val epoch time:   {:s}".format(
            rank,
            src.utils.time.sprint_fancy_time_diff(val_epoch_time)
        ))

        torch.distributed.barrier()

        logger.print("    [Rank {:d}] Val mean loss:    {:.2e}".format(
            rank,
            val_mean_loss
        ))

        torch.distributed.barrier()


        ## Scheduler update

        if rank == 0:
            exp_data["results"]["stage_1"]["learning_rate_list"].append(scheduler.get_last_lr()[0])

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
            logger.print(src.utils.nvgpu.sprint_memory_usage(exp_params["settings"]["device_idxs"], num_spaces=6))
        

        ## Training conditions and checkpoints

        if rank == 0:

            ### Errors

            if math.isnan(sum(list(train_mean_loss_list_mp))) or math.isnan(sum(list(val_mean_loss_list_mp))):
                finished_mp.value = True

            ### Number of epochs

            if num_epochs >= max_epochs:
                finished_mp.value = True

            ### Early stopping

            if early_stopper.early_stop(sum(val_mean_loss_list_mp)):
                finished_mp.value = True

            ### Checkpoint saving

            with src.utils.signal.DelayedInterrupt():

                experiment_checkpoint = {}

                experiment_checkpoint["backbone_state_dict"] = backbone.state_dict()
                experiment_checkpoint["ret_head_state_dict"] = ret_head.state_dict()
                experiment_checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                experiment_checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                experiment_checkpoint["early_stopper_state_dict"] = early_stopper.state_dict()
                experiment_checkpoint["best_tracker_state_dict"] = best_tracker.state_dict()
                if grad_scaler is not None:
                    experiment_checkpoint["grad_scaler_state_dict"] = grad_scaler.state_dict()

                ### Save best component checkpoint

                if best_tracker.is_best(sum(train_mean_loss_list_mp)):

                    best_stage_1_experiment_checkpoint_filename = os.path.join(
                        experiment_dirname, "train_ret_DDP_stage_1__best_ckp.pth"
                    )

                    logger.print("    [Rank {:d}] Saving currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_1_experiment_checkpoint_filename
                    ))

                    torch.save(
                        experiment_checkpoint,
                        best_stage_1_experiment_checkpoint_filename
                    )

                    logger.print("    [Rank {:d}] Saved currently best model to \"{:s}\"".format(
                        rank,
                        best_stage_1_experiment_checkpoint_filename
                    ))
                    
                ### Save experiment data

                exp_data["results"]["stage_1"]["num_epochs"] = num_epochs
                exp_data["results"]["stage_1"]["finished"] = finished_mp.value

                exp_data_filename = os.path.join(
                    experiment_dirname, "train_ret_DDP_stage_1__data.json"
                )

                logger.print("    [Rank {:d}] Saving experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                src.utils.json.save_json_dict(
                    exp_data,
                    exp_data_filename
                )

                logger.print("    [Rank {:d}] Saved experiment data to \"{:s}\"".format(
                    rank,
                    exp_data_filename
                ))

                ### Save last component checkpoint
                
                last_stage_1_experiment_checkpoint_filename = os.path.join(
                    experiment_dirname, "train_ret_DDP_stage_1__last_ckp.pth"
                )

                logger.print("    [Rank {:d}] Saving last epoch model to \"{:s}\"".format(
                    rank,
                    last_stage_1_experiment_checkpoint_filename
                ))

                torch.save(
                    experiment_checkpoint,
                    last_stage_1_experiment_checkpoint_filename
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



########
# MAIN SCRIPT
########



if __name__ == "__main__":


    datetime_now_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")


    ####
    # COMMAND ARGUMENTS
    ####


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "exp_params_filename",
        help="filename of the experiment params json file"
    )
    
    parser.add_argument(
        "--master_port", 
        help="master port for multi-GPU communication",
        type=int, default=12355
    )
    
    parser.add_argument(
        "--terminal_silent",
        help="do not make prints to the terminal",
        action="store_true"
    )
    
    parser.add_argument(
        "--disable_tqdm",
        help="do not print nor log tqdm bars",
        action="store_true"
    )

    parser.add_argument(
        "--resume_experiment",
        help="ignores finished flag in experiment data, necessary for resuming experiment",
        action="store_true"
    )
    
    parser.add_argument(
        "--reset_experiment",
        help="deletes experiment data directory, necessary for restarting experiment",
        action="store_true"
    )
    
    parser.add_argument(
        "--autograd_anomaly",
        help="use torch.autograd.set_detect_anomaly(True) for debugging",
        action="store_true"
    )
    
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

    log_filename = "train_ret_DDP_stage_1__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = src.utils.log.Logger(logger_streams)

    sys.stderr = logger
    
    logger.print("Bash command:", " ".join(sys.argv))


    ####
    # GPU INITIALIZATION
    ####
    

    logger.print("Initializing GPU devices")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_idxs = exp_params["settings"]["device_idxs"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in device_idxs])

    torch.cuda.empty_cache()


    ####
    # PREPARE EXPERIMENT DATA
    ####
    
    
    logger.print("Preparing experiment data")

    exp_data_filename = os.path.join(
        experiment_dirname, "train_ret_DDP_stage_1__data.json"
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
        exp_data["settings"]["head"] = exp_params["settings"]["head"]

        exp_data["settings"]["command_args"] = vars(command_args)

        exp_data["settings"]["gpu_devices"] = src.utils.nvgpu.list_gpu_data(device_idxs)
        exp_data["settings"]["hostname"] = socket.gethostname()
        exp_data["settings"]["master_port"] = command_args.master_port

        exp_data_filename = os.path.join(
            experiment_dirname, "train_ret_DDP_stage_1__data.json"
        )
        
        src.utils.json.save_json_dict(
            exp_data,
            exp_data_filename
        )

        logger.print("Starting experiment at {:s}".format(datetime_now_str))

    else:

        logger.print("Resuming experiment at {:s}".format(datetime_now_str))
    
    exp_data = src.utils.json.load_json_dict(exp_data_filename)


    ####
    # STAGE 1 - FROZEN BACKBONE
    ####


    # Initialize experiment data
    
    
    logger.print("Initializing experiment data")

    ## Settings

    exp_data["settings"]["stage_1"] = {}
    exp_data["settings"]["stage_1"].update(exp_params["settings"]["stage_1"])
    
    stage_description = "Train with frozen backbone"
    exp_data["settings"]["stage_1"]["description"] = stage_description

    if src.utils.dict.chain_get(exp_data, "results", "stage_1") is None:

        ## Results

        exp_data["results"]["stage_1"] = {}

        exp_data["results"]["stage_1"]["train_epoch_time_list"] = []
        exp_data["results"]["stage_1"]["val_epoch_time_list"] = []

        exp_data["results"]["stage_1"]["train_mean_loss_list"] = []
        exp_data["results"]["stage_1"]["val_mean_loss_list"] = []

        exp_data["results"]["stage_1"]["learning_rate_list"] = []

        exp_data["results"]["stage_1"]["num_epochs"] = 0
        exp_data["results"]["stage_1"]["finished"] = False
        
        src.utils.json.save_json_dict(
            exp_data,
            exp_data_filename
        )


    # Check if stage is finished

    finished = exp_data["results"]["stage_1"]["finished"] and not command_args.resume_experiment

    if not finished:

        world_size = len(exp_params["settings"]["device_idxs"])
        manager = torch.multiprocessing.Manager()

        train_epoch_time_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        val_epoch_time_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        train_mean_loss_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        val_mean_loss_list_mp = manager.Array("f", [0 for _ in range(world_size)])
        dataset_random_seed_mp = manager.Value("i", 0)
        dataset_random_mode_mp = manager.Value("s", 0)
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
                dataset_random_mode_mp,
                finished_mp
            ],
            nprocs=world_size
        )
