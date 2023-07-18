import os
import sys
import pathlib
import pickle as pkl

import numpy as np

import torch
import torchvision

from torch.utils.data import DataLoader, Subset

from datasets import deep_fashion
from arch import backbones, models, heads

from tqdm import tqdm

import utils.mem, utils.list, utils.training, utils.time, utils.log

from time import time
from datetime import datetime


########


def save_checkpoint(filename):

    checkpoint = {
        "model_state_dict": model.module.state_dict()
        }

    torch.save(checkpoint, filename)


def load_checkpoint(filename):
    
    checkpoint = torch.load(filename)

    # Loading model

    backbone = backbones.ResNet50Backbone()

    model = models.RetModel(backbone, 1024).to(first_device)
    model = torch.nn.DataParallel(model, device_ids=list(range(len(device_idxs))))
    model.module.load_state_dict(checkpoint["model_state_dict"])

    return model


def save_train_data(filename):

    with open(filename, 'wb') as file:
        pkl.dump(train_data, file)


def train_epoch(data_loader, with_tqdm=True):

    model.train()
    total_loss = 0

    loader_gen = data_loader
    if with_tqdm: loader_gen = tqdm(loader_gen)

    for batch in loader_gen:

        anc_imgs = batch[0]
        pos_imgs = batch[1]
        neg_imgs = batch[2]

        with torch.cuda.amp.autocast():

            anc_emb = model(anc_imgs)
            pos_emb = model(pos_imgs)
            neg_emb = model(neg_imgs)
            
            triplet_loss = torch.nn.TripletMarginLoss()
            loss = triplet_loss(anc_emb, pos_emb, neg_emb)

        total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return total_loss


def eval_epoch(data_loader, with_tqdm=True):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        loader_gen = data_loader
        if with_tqdm: loader_gen = tqdm(loader_gen)

        for batch in loader_gen:

            anc_imgs = batch[0]
            pos_imgs = batch[1]
            neg_imgs = batch[2]

            with torch.cuda.amp.autocast():

                anc_emb = model(anc_imgs)
                pos_emb = model(pos_imgs)
                neg_emb = model(neg_imgs)

                triplet_loss = torch.nn.TripletMarginLoss()
                loss = triplet_loss(anc_emb, pos_emb, neg_emb)

            total_loss += loss.item()

    return total_loss


########


if __name__ == "__main__":

    # Training settings

    training_name_str = "convnext_tiny_ret"
    now_datetime_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")

    results_dir = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", training_name_str + "_" + now_datetime_str)
    os.mkdir(results_dir)

    train_data = {}
    train_data["settings"] = {}
    train_data["results"] = {}

    ## Prepare logs

    log_filename = "log.txt"
    log_full_filename = os.path.join(results_dir, log_filename)

    logger = utils.log.Logger([
        sys.stdout,
        log_full_filename
    ])

    ## Select CUDA devices

    logger.print("Select CUDA devices")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_idxs = [3, 4, 5, 6]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in device_idxs])

    first_device = torch.device("cuda:0")
    device = torch.device("cuda")

    train_data["settings"]["gpu_usage"] = utils.mem.list_gpu_data(device_idxs)

    ## Release unallocated GPU memory

    logger.print("Releasing unallocated GPU memory")

    torch.cuda.empty_cache()

    logger.print("Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=2))

    ## Create datasets

    logger.print("Create datasets")

    ctsrbm_image_transform = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    ctsrbm_image_transform.antialias = True

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")

    ctsrbm_dataset = deep_fashion.ConsToShopClothRetrBM(ctsrbm_dataset_dir, ctsrbm_image_transform)

    cutdown_ratio = 1

    if cutdown_ratio == 1:
        ctsrbm_train_dataset = Subset(ctsrbm_dataset, ctsrbm_dataset.get_split_mask_idxs("train"))
        ctsrbm_test_dataset = Subset(ctsrbm_dataset, ctsrbm_dataset.get_split_mask_idxs("test"))
        ctsrbm_val_dataset = Subset(ctsrbm_dataset, ctsrbm_dataset.get_split_mask_idxs("val"))
    else:
        ctsrbm_train_dataset = Subset(ctsrbm_dataset, utils.list.cutdown_list(ctsrbm_dataset.get_split_mask_idxs("train"), cutdown_ratio))
        ctsrbm_test_dataset = Subset(ctsrbm_dataset, utils.list.cutdown_list(ctsrbm_dataset.get_split_mask_idxs("test"), cutdown_ratio))
        ctsrbm_val_dataset = Subset(ctsrbm_dataset, utils.list.cutdown_list(ctsrbm_dataset.get_split_mask_idxs("val"), cutdown_ratio))

    train_data["settings"]["datasets"] = [
        "DeepFashion Consumer-to-shop Clothes Retrieval Benchmark"
    ]

    ## Create data loaders

    logger.print("Create data loaders")

    batch_size = 256
    num_workers = 16

    train_loader = DataLoader(ctsrbm_train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(ctsrbm_test_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(ctsrbm_val_dataset, batch_size=batch_size, num_workers=num_workers)

    train_data["settings"]["data_loading"] = {
        "batch_size": batch_size,
        "num_workers": num_workers
    }

    ## Create model and training settings

    logger.print("Create model and training settings")

    backbone = backbones.ConvNeXTTinyBackbone()
    model = models.RetModel(backbone, 1024).to(first_device)
    model = torch.nn.DataParallel(model, device_ids=list(range(len(device_idxs))))

    train_data["settings"]["model"] = {
        "backbone": "ConvNeXTTinyBackbone",
        "heads": [
            "Image retrieval"
        ]
    }

    # Stage 1 - Train with frozen backbone

    logger.print("Stage 1 - Train with frozen backbone")

    train_data["settings"]["stage_1"] = {}

    train_data["settings"]["stage_1"]["description"] =\
        "Train with frozen backbone"
    
    train_data["settings"]["stage_1"]["learning_rate_list"] = []

    train_data["results"]["stage_1"] = {}

    train_data["results"]["stage_1"]["mean_train_loss_list"] = []
    train_data["results"]["stage_1"]["mean_val_loss_list"] = []
    train_data["results"]["stage_1"]["train_epoch_time_list"] = []
    train_data["results"]["stage_1"]["val_epoch_time_list"] = []

    ## Training settings

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    train_data["settings"]["stage_1"]["optimizer"] = {
        "method": "Adam",
        "learning_rate": 1e-3
    }

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )

    train_data["settings"]["stage_1"]["scheduler"] = {
        "method": "ExponentialLR",
        "gamma": 0.95
    }

    early_stopper = utils.training.EarlyStopper(patience=10)

    train_data["settings"]["stage_1"]["early_stopping"] = {
        "patience": 10
    }

    scaler = torch.cuda.amp.GradScaler()

    train_data["settings"]["stage_1"]["extra"] = [
        "automatic_mixed_precision"
    ]

    current_epoch = 0
    max_epoch = current_epoch + 30

    train_data["settings"]["stage_1"]["max_epochs"] = 30

    best_tracker = utils.training.BestTracker()
    
    ## Begin training

    model.module.freeze_backbone()

    while current_epoch < max_epoch:

        current_epoch += 1

        logger.print("Epoch {:d}".format(current_epoch))

        train_data["settings"]["stage_1"]["learning_rate_list"].append(scheduler.get_last_lr())

        start_time = time()
        train_loss = train_epoch(train_loader) 
        end_time = time()

        train_data["results"]["stage_1"]["train_epoch_time_list"].append(end_time - start_time)

        logger.print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))

        start_time = time()
        val_loss = eval_epoch(val_loader)
        end_time = time()

        train_data["results"]["stage_1"]["val_epoch_time_list"].append(end_time - start_time)

        logger.print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))

        mean_train_loss = train_loss / len(ctsrbm_train_dataset)
        mean_val_loss = val_loss / len(ctsrbm_val_dataset)

        train_data["results"]["stage_1"]["mean_train_loss_list"].append(mean_train_loss)
        train_data["results"]["stage_1"]["mean_val_loss_list"].append(mean_val_loss)

        logger.print("  Mean train loss: {:.2e}".format(mean_train_loss))
        logger.print("  Mean val loss:   {:.2e}".format(mean_val_loss))
            
        logger.print("  Current memory usage:")
        logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))

        if early_stopper.early_stop(mean_val_loss):

            break

        if best_tracker.is_best(mean_val_loss):

            logger.print("  Save stage 1 model checkpoint (best)")

            checkpoint_filename = "stage1_best_ckp.pth"
            checkpoint_full_filename = os.path.join(results_dir, checkpoint_filename)

            save_checkpoint(checkpoint_full_filename)

            logger.print("    Saved to " + checkpoint_full_filename)

        scheduler.step()

    train_data["results"]["stage_1"]["num_epochs"] = current_epoch

    ## Save stage 1 model checkpoint

    logger.print("Save stage 1 model checkpoint")

    checkpoint_filename = "stage1_last_ckp.pth"
    checkpoint_full_filename = os.path.join(results_dir, checkpoint_filename)

    save_checkpoint(checkpoint_full_filename)

    logger.print("  Saved to " + checkpoint_full_filename)

    # Stage 2 - Train entire model

    logger.print("Stage 2 - Train entire model")

    train_data["settings"]["stage_2"] = {}

    train_data["settings"]["stage_2"]["description"] =\
        "Train entire model"
    
    train_data["settings"]["stage_2"]["learning_rate_list"] = []

    train_data["results"]["stage_2"] = {}

    train_data["results"]["stage_2"]["mean_train_loss_list"] = []
    train_data["results"]["stage_2"]["mean_val_loss_list"] = []
    train_data["results"]["stage_2"]["train_epoch_time_list"] = []
    train_data["results"]["stage_2"]["val_epoch_time_list"] = []

    ## Training settings

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    train_data["settings"]["stage_2"]["optimizer"] = {
        "method": "Adam",
        "learning_rate": 1e-3
    }

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )

    train_data["settings"]["stage_2"]["scheduler"] = {
        "method": "ExponentialLR",
        "gamma": 0.95
    }

    early_stopper = utils.training.EarlyStopper(patience=10)

    train_data["settings"]["stage_2"]["early_stopping"] = {
        "patience": 10
    }

    scaler = torch.cuda.amp.GradScaler()

    train_data["settings"]["stage_2"]["extra"] = [
        "automatic_mixed_precision"
    ]

    max_epoch = current_epoch + 30

    train_data["settings"]["stage_2"]["max_epochs"] = 30

    best_tracker = utils.training.BestTracker()

    ## Load stage 1 model checkpoint (best)

    logger.print("Load stage 1 model checkpoint (best)")

    checkpoint_filename = "stage1_best_ckp.pth"
    checkpoint_full_filename = os.path.join(results_dir, checkpoint_filename)

    model = load_checkpoint(checkpoint_full_filename)

    logger.print("  Loaded")
    
    ## Begin training

    model.module.unfreeze_backbone()
    
    while current_epoch < max_epoch:

        current_epoch += 1

        logger.print("Epoch {:d}".format(current_epoch))

        train_data["settings"]["stage_2"]["learning_rate_list"].append(scheduler.get_last_lr())

        start_time = time()
        train_loss = train_epoch(train_loader) 
        end_time = time()

        train_data["results"]["stage_2"]["train_epoch_time_list"].append(end_time - start_time)

        logger.print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))

        start_time = time()
        val_loss = eval_epoch(val_loader)
        end_time = time()

        train_data["results"]["stage_2"]["val_epoch_time_list"].append(end_time - start_time)

        logger.print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))

        mean_train_loss = train_loss / len(ctsrbm_train_dataset)
        mean_val_loss = val_loss / len(ctsrbm_val_dataset)

        train_data["results"]["stage_2"]["mean_train_loss_list"].append(mean_train_loss)
        train_data["results"]["stage_2"]["mean_val_loss_list"].append(mean_val_loss)

        logger.print("  Mean train loss: {:.2e}".format(mean_train_loss))
        logger.print("  Mean val loss:   {:.2e}".format(mean_val_loss))
            
        logger.print("  Current memory usage:")
        logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))

        if early_stopper.early_stop(mean_val_loss):

            break

        if best_tracker.is_best(mean_val_loss):

            logger.print("  Save stage 2 model checkpoint (best)")

            checkpoint_filename = "stage2_best_ckp.pth"
            checkpoint_full_filename = os.path.join(results_dir, checkpoint_filename)

            save_checkpoint(checkpoint_full_filename)

            logger.print("    Saved to " + checkpoint_full_filename)

        scheduler.step()

    train_data["results"]["stage_2"]["num_epochs"] =\
        current_epoch - train_data["results"]["stage_1"]["num_epochs"]

    ## Save stage 2 model checkpoint

    logger.print("Save stage 2 model checkpoint")

    checkpoint_filename = "stage2_last_ckp.pth"
    checkpoint_full_filename = os.path.join(results_dir, checkpoint_filename)

    save_checkpoint(checkpoint_full_filename)

    logger.print("  Saved to " + checkpoint_full_filename)

    # Final test

    test_loss = eval_epoch(test_loader)
    train_data["test_loss"] = test_loss

    # Save training data

    logger.print("Save train data")

    train_data_filename = "data.pkl"
    train_data_full_filename = os.path.join(results_dir, train_data_filename)

    save_train_data(train_data_full_filename)

    logger.print("  Saved to " + train_data_full_filename)
