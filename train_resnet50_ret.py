#!.venv/bin/python3

import os
import sys
import pathlib

import numpy as np

import torch
import torchvision

from torch.utils.data import DataLoader, Subset

from datasets import deep_fashion
from arch import backbones, heads, models

from tqdm import tqdm

import utils.mem, utils.list, utils.training

from time import time
from datetime import datetime


########


def save_checkpoint(checkpoint_filename):

    checkpoint = {
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_data": train_data
        }

    torch.save(checkpoint, checkpoint_filename)


def load_checkpoint(checkpoint_filename):
    
    checkpoint = torch.load(checkpoint_filename)

    # Loading model

    backbone = backbones.ResNet50Backbone()

    model = models.RetModel(backbone, 1024).to(first_device)
    model = torch.nn.DataParallel(model, device_ids=list(range(len(device_idxs))))
    model.module.load_state_dict(checkpoint["model_state_dict"])
    
    # Loading optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Loading scheduler

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )

    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Loading other parameters

    train_data = checkpoint["train_data"]

    return model, optimizer, scheduler, train_data


def train_epoch(with_tqdm=True):

    model.train()
    total_loss = 0

    loader_gen = ctsrbm_train_loader
    if with_tqdm: loader_gen = tqdm(loader_gen)

    for train_batch in loader_gen:

        anc_imgs = train_batch[0]
        pos_imgs = train_batch[1]
        neg_imgs = train_batch[2]

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


def val_epoch(with_tqdm=True):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        loader_gen = ctsrbm_val_loader
        if with_tqdm: loader_gen = tqdm(loader_gen)

        for val_batch in loader_gen:

            anc_imgs = val_batch[0]
            pos_imgs = val_batch[1]
            neg_imgs = val_batch[2]

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

    # Training variables

    training_name_str = "resnet50_ret"
    now_datetime_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")

    # Define log file

    log_dir = os.path.join(pathlib.Path.home(), "data", "logs", "fashion_retrieval")
    log_filename = training_name_str + "_log_" + now_datetime_str + ".txt"
    log_full_filename = os.path.join(log_dir, log_filename)

    log_file = open(log_full_filename, 'w')

    # Select CUDA devices

    print("Select CUDA devices")
    print("Select CUDA devices", file=log_file)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_idxs = [3, 4, 5, 6]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in device_idxs])

    first_device = torch.device("cuda:0")
    device = torch.device("cuda")

    # Release unallocated GPU memory

    print("Releasing unallocated GPU memory")
    print("Releasing unallocated GPU memory", file=log_file)

    torch.cuda.empty_cache()

    print("Current memory usage:")
    print("Current memory usage:", file=log_file)

    print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=2))
    print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=2), file=log_file)

    # Create datasets

    print("Create datasets")
    print("Create datasets", file=log_file)

    ctsrbm_image_transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
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

    # Create data loaders

    print("Create data loaders")
    print("Create data loaders", file=log_file)

    batch_size = 256
    num_workers = 16

    ctsrbm_train_loader = DataLoader(ctsrbm_train_dataset, batch_size=batch_size, num_workers=num_workers)
    ctsrbm_test_loader = DataLoader(ctsrbm_test_dataset, batch_size=batch_size, num_workers=num_workers)
    ctsrbm_val_loader = DataLoader(ctsrbm_val_dataset, batch_size=batch_size, num_workers=num_workers)

    # Create model and training settings

    print("Create model and training settings")
    print("Create model and training settings", file=log_file)

    backbone = backbones.ResNet50Backbone()
    model = models.RetModel(backbone, 1024).to(first_device)
    model = torch.nn.DataParallel(model, device_ids=list(range(len(device_idxs))))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )

    scaler = torch.cuda.amp.GradScaler()

    # Initialize training

    print("Initialize training")
    print("Initialize training", file=log_file)

    train_data = {}

    train_data["mean_train_loss_list"] = []
    train_data["mean_val_loss_list"] = []
    train_data["train_epoch_time_list"] = []
    train_data["val_epoch_time_list"] = []

    current_epoch = 0

    # Train with frozen backbone

    print("Train with frozen backbone")
    print("Train with frozen backbone", file=log_file)

    model.module.freeze_backbone()
    early_stopper = utils.EarlyStopper(patience=10)
    max_epoch = current_epoch + 30

    first_epoch = True 

    while current_epoch < max_epoch:

        current_epoch += 1

        print("  Epoch {:d}".format(current_epoch))
        print("  Epoch {:d}".format(current_epoch), file=log_file)

        start_time = time()
        train_loss = train_epoch() 
        end_time = time()

        train_data["train_epoch_time_list"].append(end_time - start_time)

        print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))
        print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)), file=log_file)

        start_time = time()
        val_loss = val_epoch()
        end_time = time()

        train_data["val_epoch_time_list"].append(end_time - start_time)

        print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))
        print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)), file=log_file)

        print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))
        print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4), file=log_file)

        mean_train_loss = train_loss / len(ctsrbm_train_dataset)
        mean_val_loss = val_loss / len(ctsrbm_val_dataset)

        train_data["mean_train_loss_list"].append(mean_train_loss)
        train_data["mean_val_loss_list"].append(mean_val_loss)

        print("  Mean train loss: {:.2e}".format(mean_train_loss))
        print("  Mean train loss: {:.2e}".format(mean_train_loss), file=log_file)

        print("  Mean val loss:   {:.2e}".format(mean_val_loss))
        print("  Mean val loss:   {:.2e}".format(mean_val_loss), file=log_file)

        if first_epoch:
            first_epoch = False
            print("  Current memory usage:")
            print("  Current memory usage:", file=log_file)

        if early_stopper.early_stop(mean_val_loss):
            break

        scheduler.step()

    train_data["stage_1_epochs"] = current_epoch

    # Save stage 1 model checkpoint

    print("Save stage 1 model checkpoint")
    print("Save stage 1 model checkpoint", file=log_file)

    checkpoint_dir = os.path.join(pathlib.Path.home(), "data", "checkpoints", "fashion_retrieval")
    checkpoint_filename = training_name_str + "_stage1_" + now_datetime_str + ".pth"
    checkpoint_full_filename = os.path.join(checkpoint_dir, checkpoint_filename)

    save_checkpoint(checkpoint_full_filename)

    print("  Saved to " + checkpoint_full_filename)
    print("  Saved to " + checkpoint_full_filename, file=log_file)

    # Train entire model

    print("Train entire model")
    print("Train entire model", file=log_file)

    model.module.unfreeze_backbone()
    early_stopper = utils.training.EarlyStopper(patience=10)
    max_epoch = current_epoch + 30

    first_epoch = True 

    while current_epoch < max_epoch:

        current_epoch += 1

        print("  Epoch {:d}".format(current_epoch))
        print("  Epoch {:d}".format(current_epoch), file=log_file)

        start_time = time()
        train_loss = train_epoch() 
        end_time = time()

        train_data["train_epoch_time_list"].append(end_time - start_time)

        print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))
        print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)), file=log_file)

        start_time = time()
        val_loss = val_epoch()
        end_time = time()

        train_data["val_epoch_time_list"].append(end_time - start_time)

        print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)))
        print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(end_time - start_time)), file=log_file)

        print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))
        print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4), file=log_file)

        mean_train_loss = train_loss / len(ctsrbm_train_dataset)
        mean_val_loss = val_loss / len(ctsrbm_val_dataset)

        train_data["mean_train_loss_list"].append(mean_train_loss)
        train_data["mean_val_loss_list"].append(mean_val_loss)

        print("  Mean train loss: {:.2e}".format(mean_train_loss))
        print("  Mean train loss: {:.2e}".format(mean_train_loss), file=log_file)
        
        print("  Mean val loss:   {:.2e}".format(mean_val_loss))
        print("  Mean val loss:   {:.2e}".format(mean_val_loss), file=log_file)

        if first_epoch:
            first_epoch = False
            print("  Current memory usage:")
            print("  Current memory usage:", file=log_file)

        if early_stopper.early_stop(mean_val_loss):
            break

        scheduler.step()

    train_data["stage_2_epochs"] = current_epoch - train_data["stage_1_epochs"]

    # Save stage 2 model checkpoint

    print("Save stage 2 model checkpoint")
    print("Save stage 2 model checkpoint", file=log_file)

    checkpoint_dir = os.path.join(pathlib.Path.home(), "data", "checkpoints", "fashion_retrieval")
    checkpoint_filename = training_name_str + "_stage2_" + now_datetime_str + ".pth"
    checkpoint_full_filename = os.path.join(checkpoint_dir, checkpoint_filename)

    save_checkpoint(checkpoint_full_filename)

    print("  Saved to " + checkpoint_full_filename)
    print("  Saved to " + checkpoint_full_filename, file=log_file)
