import os
import sys
import pathlib
import pickle as pkl

import numpy as np

import torch
import torchvision

from torch.utils.data import DataLoader, Subset

from datasets import deep_fashion, fusion
from arch import backbones, models, heads

from tqdm import tqdm

import utils.mem, utils.list, fashion_retrieval.utils.train, utils.time, utils.log

from time import time
from datetime import datetime

from itertools import chain
from functools import reduce

import json


########


def save_checkpoint(filename):

    checkpoint = {
        "backbone_state_dict": backbone.state_dict(),
        "cls_head_state_dict": cls_head.state_dict(),
        "ret_head_state_dict": ret_head.state_dict()
        }

    torch.save(checkpoint, filename)


def load_checkpoint(filename):
    
    checkpoint = torch.load(filename)

    backbone = backbones.ResNet50Backbone()
    cls_head = heads.ClsHead(backbone.out_shape, 50)
    ret_head = heads.RetHead(backbone.out_shape, 1024)

    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    cls_head.load_state_dict(checkpoint["cls_head_state_dict"])
    ret_head.load_state_dict(checkpoint["ret_head_state_dict"])

    return backbone, cls_head, ret_head


def save_train_data(filename):

    with open(filename, 'wb') as file:
        pkl.dump(train_data, file)


def create_optimizer(stage, optimizer_params):

    optimizer_class = run_params["settings"][stage]["optimizer"]["class"]

    if optimizer_class == "Adam":

        lr = run_params["settings"][stage]["optimizer"]["lr"]

        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=lr
        )

        train_data["settings"][stage]["optimizer"] = {
            "class": "Adam",
            "lr": lr
        }

    elif optimizer_class == "SGD":

        lr = run_params["settings"][stage]["optimizer"]["lr"]
        momentum = run_params["settings"][stage]["optimizer"]["momentum"]

        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=lr,
            momentum=momentum
        )

        train_data["settings"][stage]["optimizer"] = {
            "class": "SGD",
            "lr": lr,
            "momentum": momentum
        }

    else:

        msg = "Invalid [\"settings\"][\"{:s}\"][\"optimizer\"][\"class\"]".format(stage)

        logger.print(msg)
        raise ValueError(msg)
    
    return optimizer


def create_scheduler(stage, optimizer):

    scheduler_class = run_params["settings"][stage]["scheduler"]["class"]

    if scheduler_class == "ExponentialLR":

        gamma = run_params["settings"][stage]["scheduler"]["gamma"]

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

        train_data["settings"][stage]["scheduler"] = {
            "class": "ExponentialLR",
            "gamma": gamma
        }

    else:

        msg = "Invalid [\"settings\"][\"{:s}\"][\"scheduler\"][\"class\"]".format(stage)

        logger.print(msg)
        raise ValueError(msg)

    return scheduler


def create_early_stopper(stage):

    patience = run_params["settings"][stage]["early_stopper"]["patience"]
    min_delta = run_params["settings"][stage]["early_stopper"]["min_delta"]

    early_stopper = utils.train.EarlyStopper(
        patience=patience,
        min_delta=min_delta
    )

    train_data["settings"][stage]["early_stopping"] = {
        "patience": patience,
        "min_delta": min_delta
    }

    return early_stopper


def train_epoch(data_loader, with_tqdm=True):

    backbone.train()
    ret_head.train()
    cls_head.train()

    total_loss_item = 0
    total_task_loss_item_list = [0] * 2

    loader_gen = data_loader
    if with_tqdm: loader_gen = tqdm(loader_gen)

    for batch in loader_gen:

        batch_losses = batch_evaluation(batch)        
        total_batch_loss = reduce(torch.add, map(lambda item: item[1], batch_losses))

        total_loss_item += total_batch_loss.item()
        for idx, loss in batch_losses: total_task_loss_item_list[idx] += loss.item()
            
        scaler.scale(total_batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return total_loss_item, total_task_loss_item_list


def eval_epoch(data_loader, with_tqdm=True):

    backbone.eval()
    ret_head.eval()
    cls_head.eval()

    total_loss_item = 0
    total_task_loss_item_list = [0] * 2

    with torch.no_grad():

        loader_gen = data_loader
        if with_tqdm: loader_gen = tqdm(loader_gen)

        for batch in loader_gen:

            batch_losses = batch_evaluation(batch)
            total_batch_loss = reduce(torch.add, map(lambda item: item[1], batch_losses))

            total_loss_item += total_batch_loss.item()
            for idx, loss in batch_losses: total_task_loss_item_list[idx] += loss.item()

    return total_loss_item, total_task_loss_item_list


def batch_evaluation(batch):

    batch_losses = []

    for idx, sub_batch in batch:

        if idx == 0:

            imgs = sub_batch[0].to(device)
            cats = sub_batch[1].to(device)

            with torch.cuda.amp.autocast():

                pred_cats = cls_model(imgs)

                ce_loss = torch.nn.CrossEntropyLoss()
                loss = ce_loss(pred_cats, cats)

            batch_losses.append((idx, loss))

        if idx == 1:

            anc_imgs = sub_batch[0].to(device)
            pos_imgs = sub_batch[1].to(device)
            neg_imgs = sub_batch[2].to(device)

            with torch.cuda.amp.autocast():

                anc_emb = ret_model(anc_imgs)
                pos_emb = ret_model(pos_imgs)
                neg_emb = ret_model(neg_imgs)
        
                triplet_loss = torch.nn.TripletMarginLoss()
                loss = triplet_loss(anc_emb, pos_emb, neg_emb)

            batch_losses.append((idx, loss))

    return batch_losses


########


if __name__ == "__main__":

    # Training settings

    training_name_str = "resnet50_cls_ret"
    now_datetime_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")

    results_dir = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", training_name_str + "_" + now_datetime_str)
    os.mkdir(results_dir)

    train_data = {}
    train_data["settings"] = {}
    train_data["results"] = {}

    run_params_filename = os.path.join(pathlib.Path.home(), "fashion_retrieval", "run_settings", sys.argv[1])
    run_params_file = open(run_params_filename, 'r')
    run_params = json.load(run_params_file)
    run_params_file.close()

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

    device_idxs = run_params["settings"]["device_idxs"]
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

    backbone_image_transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
    backbone_image_transform.antialias = True

    capbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Category and Attribute Prediction Benchmark")
    capbm_dataset = deep_fashion.CatAttrPredBM(capbm_dataset_dir, backbone_image_transform)

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")
    ctsrbm_dataset = deep_fashion.ConsToShopClothRetrBM(ctsrbm_dataset_dir, backbone_image_transform)

    cutdown_ratio = 1

    if cutdown_ratio == 1:

        capbm_train_dataset = Subset(capbm_dataset, capbm_dataset.get_split_mask_idxs("train"))
        capbm_test_dataset = Subset(capbm_dataset, capbm_dataset.get_split_mask_idxs("test"))
        capbm_val_dataset = Subset(capbm_dataset, capbm_dataset.get_split_mask_idxs("val"))

        ctsrbm_train_dataset = Subset(ctsrbm_dataset, ctsrbm_dataset.get_split_mask_idxs("train"))
        ctsrbm_test_dataset = Subset(ctsrbm_dataset, ctsrbm_dataset.get_split_mask_idxs("test"))
        ctsrbm_val_dataset = Subset(ctsrbm_dataset, ctsrbm_dataset.get_split_mask_idxs("val"))
    
    else:

        capbm_train_dataset = Subset(capbm_dataset, utils.list.cutdown_list(capbm_dataset.get_split_mask_idxs("train"), cutdown_ratio))
        capbm_test_dataset = Subset(capbm_dataset, utils.list.cutdown_list(capbm_dataset.get_split_mask_idxs("test"), cutdown_ratio))
        capbm_val_dataset = Subset(capbm_dataset, utils.list.cutdown_list(capbm_dataset.get_split_mask_idxs("val"), cutdown_ratio))
    
        ctsrbm_train_dataset = Subset(ctsrbm_dataset, utils.list.cutdown_list(ctsrbm_dataset.get_split_mask_idxs("train"), cutdown_ratio))
        ctsrbm_test_dataset = Subset(ctsrbm_dataset, utils.list.cutdown_list(ctsrbm_dataset.get_split_mask_idxs("test"), cutdown_ratio))
        ctsrbm_val_dataset = Subset(ctsrbm_dataset, utils.list.cutdown_list(ctsrbm_dataset.get_split_mask_idxs("val"), cutdown_ratio))

    mixed_train_dataset = fusion.Fusion([capbm_train_dataset, ctsrbm_train_dataset])
    mixed_test_dataset = fusion.Fusion([capbm_test_dataset, ctsrbm_test_dataset])
    mixed_val_dataset = fusion.Fusion([capbm_val_dataset, ctsrbm_val_dataset])

    train_data["settings"]["datasets"] = [
        "DeepFashion Category and Attribute Prediction Benchmark"
        "DeepFashion Consumer-to-shop Clothes Retrieval Benchmark"
    ]

    ## Create data loaders

    logger.print("Create data loaders")

    batch_size = run_params["settings"]["data_loading"]["batch_size"]
    num_workers = run_params["settings"]["data_loading"]["num_workers"]

    train_loader = DataLoader(mixed_train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=fusion.Fusion.collate_fn)
    test_loader = DataLoader(mixed_test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=fusion.Fusion.collate_fn)
    val_loader = DataLoader(mixed_val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=fusion.Fusion.collate_fn)

    train_data["settings"]["data_loading"] = {
        "batch_size": batch_size,
        "num_workers": num_workers
    }

    ## Create model and training settings

    logger.print("Create model and training settings")

    backbone = backbones.ResNet50Backbone().to(first_device)
    
    cls_head = heads.ClsHead(backbone.out_shape, 50).to(first_device)
    ret_head = heads.RetHead(backbone.out_shape, 1024).to(first_device)

    cls_model = models.BackboneAndHead(backbone, cls_head).to(first_device)
    if len(device_idxs) > 1: cls_model = torch.nn.DataParallel(cls_model, device_ids=list(range(len(device_idxs))))

    ret_model = models.BackboneAndHead(backbone, ret_head).to(first_device)
    if len(device_idxs) > 1: ret_model = torch.nn.DataParallel(ret_model, device_ids=list(range(len(device_idxs))))

    train_data["settings"]["model"] = {
        "backbone": "ResNet50Backbone",
        "heads": [
            "RetHead",
            "ClsHead"
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
    train_data["results"]["stage_1"]["mean_cls_train_loss_list"] = []
    train_data["results"]["stage_1"]["mean_cls_val_loss_list"] = []
    train_data["results"]["stage_1"]["mean_ret_train_loss_list"] = []
    train_data["results"]["stage_1"]["mean_ret_val_loss_list"] = []
    train_data["results"]["stage_1"]["train_epoch_time_list"] = []
    train_data["results"]["stage_1"]["val_epoch_time_list"] = []

    ## Training settings

    optimizer_params = chain(
        cls_head.parameters(),
        ret_head.parameters()
    )

    optimizer = create_optimizer("stage_1", optimizer_params)

    scheduler = create_scheduler("stage_1", optimizer)

    early_stopper = create_early_stopper("stage_1")

    ### Epochs

    max_epochs = run_params["settings"]["stage_1"]["max_epochs"]

    current_epoch = 0
    max_current_epoch = current_epoch + max_epochs

    train_data["settings"]["stage_1"]["max_epochs"] = max_epochs

    ### Other

    scaler = torch.cuda.amp.GradScaler()

    train_data["settings"]["stage_1"]["extra"] = [
        "automatic_mixed_precision"
    ]

    best_tracker = utils.train.BestTracker()
    
    ## Begin training

    for param in backbone.parameters():
        param.requires_grad = False
         
    while current_epoch < max_current_epoch:

        current_epoch += 1

        logger.print("Epoch {:d}".format(current_epoch))

        train_data["settings"]["stage_1"]["learning_rate_list"].append(scheduler.get_last_lr())

        start_time = time()
        train_loss, task_train_loss_list = train_epoch(train_loader) 
        end_time = time()
        train_epoch_time = end_time - start_time

        logger.print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(train_epoch_time)))

        start_time = time()
        val_loss, task_val_loss_list = eval_epoch(val_loader)
        end_time = time()
        val_epoch_time = end_time - start_time

        logger.print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(val_epoch_time)))

        mean_train_loss = train_loss / len(mixed_train_dataset)
        mean_val_loss = val_loss / len(mixed_val_dataset)

        logger.print("  Mean train loss: {:.2e}".format(mean_train_loss))
        logger.print("  Mean val loss:   {:.2e}".format(mean_val_loss))
            
        logger.print("  Current memory usage:")
        logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))

        mean_cls_train_loss = task_train_loss_list[0] / len(ctsrbm_train_dataset)
        mean_cls_val_loss = task_val_loss_list[0] / len(ctsrbm_train_dataset)
        mean_ret_train_loss = task_train_loss_list[1] / len(capbm_train_dataset)
        mean_ret_val_loss = task_val_loss_list[1] / len(capbm_train_dataset)

        train_data["results"]["stage_1"]["mean_train_loss_list"].append(mean_train_loss)
        train_data["results"]["stage_1"]["mean_val_loss_list"].append(mean_val_loss)
        train_data["results"]["stage_1"]["mean_cls_train_loss_list"].append(mean_cls_train_loss)
        train_data["results"]["stage_1"]["mean_cls_val_loss_list"].append(mean_cls_val_loss)
        train_data["results"]["stage_1"]["mean_ret_train_loss_list"].append(mean_ret_train_loss)
        train_data["results"]["stage_1"]["mean_ret_val_loss_list"].append(mean_ret_val_loss)
        train_data["results"]["stage_1"]["train_epoch_time_list"].append(train_epoch_time)
        train_data["results"]["stage_1"]["val_epoch_time_list"].append(val_epoch_time)

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
    train_data["results"]["stage_2"]["mean_cls_train_loss_list"] = []
    train_data["results"]["stage_2"]["mean_cls_val_loss_list"] = []
    train_data["results"]["stage_2"]["mean_ret_train_loss_list"] = []
    train_data["results"]["stage_2"]["mean_ret_val_loss_list"] = []
    train_data["results"]["stage_2"]["train_epoch_time_list"] = []
    train_data["results"]["stage_2"]["val_epoch_time_list"] = []
    
    ## Load stage 1 model checkpoint (best)

    logger.print("Load stage 1 model checkpoint (best)")

    checkpoint_filename = "stage1_best_ckp.pth"
    checkpoint_full_filename = os.path.join(results_dir, checkpoint_filename)

    backbone, cls_head, ret_head = load_checkpoint(checkpoint_full_filename)

    logger.print("  Loaded")

    cls_model = models.BackboneAndHead(backbone, cls_head).to(first_device)
    if len(device_idxs) > 1: cls_model = torch.nn.DataParallel(cls_model, device_ids=list(range(len(device_idxs))))

    ret_model = models.BackboneAndHead(backbone, ret_head).to(first_device)
    if len(device_idxs) > 1: ret_model = torch.nn.DataParallel(ret_model, device_ids=list(range(len(device_idxs))))

    ## Training settings

    optimizer_params = chain(
        backbone.parameters(),
        cls_head.parameters(),
        ret_head.parameters()
    )

    optimizer = create_optimizer("stage_2", optimizer_params)

    scheduler = create_scheduler("stage_2", optimizer)

    early_stopper = create_early_stopper("stage_2")

    ### Epochs

    max_epochs = run_params["settings"]["stage_2"]["max_epochs"]

    max_current_epoch = current_epoch + max_epochs

    train_data["settings"]["stage_2"]["max_epochs"] = max_epochs

    ### Other

    scaler = torch.cuda.amp.GradScaler()

    train_data["settings"]["stage_2"]["extra"] = [
        "automatic_mixed_precision"
    ]

    best_tracker = utils.train.BestTracker()
    
    ## Begin training

    for param in backbone.parameters():
        param.requires_grad = True
    
    while current_epoch < max_current_epoch:

        current_epoch += 1

        logger.print("Epoch {:d}".format(current_epoch))

        train_data["settings"]["stage_2"]["learning_rate_list"].append(scheduler.get_last_lr())

        start_time = time()
        train_loss, task_train_loss_list = train_epoch(train_loader) 
        end_time = time()
        train_epoch_time = end_time - start_time

        logger.print("  Train epoch time: {:s}".format(utils.time.sprint_fancy_time_diff(train_epoch_time)))

        start_time = time()
        val_loss, task_val_loss_list = eval_epoch(val_loader)
        end_time = time()
        val_epoch_time = end_time - start_time

        logger.print("  Val epoch time:   {:s}".format(utils.time.sprint_fancy_time_diff(val_epoch_time)))

        mean_train_loss = train_loss / len(mixed_train_dataset)
        mean_val_loss = val_loss / len(mixed_val_dataset)

        logger.print("  Mean train loss: {:.2e}".format(mean_train_loss))
        logger.print("  Mean val loss:   {:.2e}".format(mean_val_loss))
            
        logger.print("  Current memory usage:")
        logger.print(utils.mem.sprint_memory_usage(device_idxs, num_spaces=4))

        mean_cls_train_loss = task_train_loss_list[0] / len(ctsrbm_train_dataset)
        mean_cls_val_loss = task_val_loss_list[0] / len(ctsrbm_train_dataset)
        mean_ret_train_loss = task_train_loss_list[1] / len(capbm_train_dataset)
        mean_ret_val_loss = task_val_loss_list[1] / len(capbm_train_dataset)

        train_data["results"]["stage_2"]["mean_train_loss_list"].append(mean_train_loss)
        train_data["results"]["stage_2"]["mean_val_loss_list"].append(mean_val_loss)
        train_data["results"]["stage_2"]["mean_cls_train_loss_list"].append(mean_cls_train_loss)
        train_data["results"]["stage_2"]["mean_cls_val_loss_list"].append(mean_cls_val_loss)
        train_data["results"]["stage_2"]["mean_ret_train_loss_list"].append(mean_ret_train_loss)
        train_data["results"]["stage_2"]["mean_ret_val_loss_list"].append(mean_ret_val_loss)
        train_data["results"]["stage_2"]["train_epoch_time_list"].append(train_epoch_time)
        train_data["results"]["stage_2"]["val_epoch_time_list"].append(val_epoch_time)

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

    # Final test evaluation

    logger.print("Final test evaluation")

    test_loss, task_test_loss_list = eval_epoch(test_loader)

    mean_test_loss = test_loss / len(mixed_train_dataset)
    mean_cls_test_loss = task_test_loss_list[0] / len(ctsrbm_train_dataset)
    mean_ret_test_loss = task_test_loss_list[1] / len(capbm_train_dataset)

    logger.print("  Mean test loss: {:.2e}".format(mean_test_loss))
    logger.print("  Mean cls test loss: {:.2e}".format(mean_cls_test_loss))
    logger.print("  Mean ret test loss: {:.2e}".format(mean_ret_test_loss))

    train_data["mean_test_loss"] = mean_test_loss
    train_data["mean_cls_test_loss"] = mean_cls_test_loss
    train_data["mean_ret_test_loss"] = mean_ret_test_loss

    # Save training data

    logger.print("Save train data")

    train_data_filename = "data.pkl"
    train_data_full_filename = os.path.join(results_dir, train_data_filename)

    save_train_data(train_data_full_filename)

    logger.print("  Saved to " + train_data_full_filename)
