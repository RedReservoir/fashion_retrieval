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

from datasets import deep_fashion_ctsrbm
from arch import models, heads

from tqdm import tqdm
from fashion_retrieval.arch import backbones_cnn

import utils.mem
import utils.list
import utils.train
import utils.time
import utils.log
import utils.dict
import utils.sig
import utils.pkl
import utils.chunk
import utils.arr

from time import time
from datetime import datetime

from itertools import chain
from functools import reduce

import json
import socket



def print_tensor_info(tensor, name, logger):

    logger.print(
        "{:s}:".format(name),
        tensor.shape,
        tensor.dtype,
        tensor.device,
        utils.mem.sprint_fancy_num_bytes(utils.mem.get_num_bytes(tensor))
    )



########
# COMPONENT FUNCTIONS
########



def create_backbone(backbone_class):

    if backbone_class == "ResNet50Backbone":
        backbone = backbones_cnn.ResNet50Backbone()
    if backbone_class == "EfficientNetB3Backbone":
        backbone = backbones_cnn.EfficientNetB3Backbone()
    if backbone_class == "EfficientNetB4Backbone":
        backbone = backbones_cnn.EfficientNetB4Backbone()
    if backbone_class == "EfficientNetB5Backbone":
        backbone = backbones_cnn.EfficientNetB5Backbone()
    if backbone_class == "ConvNeXtTinyBackbone":
        backbone = backbones_cnn.ConvNeXtTinyBackbone(contiguous_after_permute=True)

    return backbone


def load_experiment_checkpoint(
        experiment_checkpoint_filename,
        exp_params,
        device
        ):

    # Load checkpoint

    experiment_checkpoint = torch.load(experiment_checkpoint_filename)

    # Backbone

    backbone_class = exp_params["settings"]["backbone"]["class"]

    backbone = create_backbone(backbone_class).to(device)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])

    # Head

    ret_head = heads.RetHead(backbone.out_shape, 1024).to(device)

    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    return (backbone, ret_head)



########
# DATASET FUNCTIONS
########



def create_backbone_transform(backbone_class):

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

    return backbone_image_transform



########
# JSON DATA FUNCTIONS
########



def save_json_data(
        json_data_filename,
        json_data
        ):

    with open(json_data_filename, 'w') as json_data_file:
        json.dump(json_data, json_data_file, indent=2)


def load_json_data(
        json_data_filename
        ):

    with open(json_data_filename, 'r') as json_data_file:
        json_data = json.load(json_data_file)

    return json_data



########
# PERFORMANCE FUNCTIONS
########



def compute_embeddings_and_item_ids(
    ret_model,
    image_loader,
    device,
    with_tqdm=True
):

    ret_model.eval()

    # Embedding calculation

    all_img_embs = torch.tensor([], dtype=float).to(device)
    all_item_ids = torch.tensor([], dtype=int).to(device)

    loader_gen = image_loader
    if with_tqdm: loader_gen = tqdm(loader_gen)

    with torch.no_grad():

        for batch in loader_gen:

            # img_embs

            imgs = batch[0].to(device)
            img_embs = ret_model(imgs)
            all_img_embs = torch.cat([all_img_embs, img_embs])

            # item_ids

            item_ids = batch[1].to(device)
            all_item_ids = torch.cat([all_item_ids, item_ids])

    return all_img_embs, all_item_ids


def compute_performance_metrics(
    shop_img_embs,
    shop_item_ids,
    cons_img_embs,
    cons_item_ids,
    k_values,
    cons_imgs_chunk_size,
    with_tqdm
):

    num_cons_imgs = cons_img_embs.shape[0]

    avg_p_at_k_dict = {k: 0 for k in k_values}
    avg_r_at_k_dict = {k: 0 for k in k_values}

    # Precision and recall metrics

    ## [i]: Number of shop items with the same item id as cons img i

    cons_shop_item_id_counts = torch.empty_like(cons_item_ids)

    for idx, cons_item_id in enumerate(cons_item_ids):
        counts = torch.sum(torch.eq(shop_item_ids, cons_item_id))
        cons_shop_item_id_counts[idx] = counts

    cons_img_idxs_chunk_gen = utils.chunk.chunk_partition_size(np.arange(num_cons_imgs), cons_imgs_chunk_size)
    if with_tqdm: cons_img_idxs_chunk_gen = tqdm(cons_img_idxs_chunk_gen)

    for cons_img_idxs_chunk in cons_img_idxs_chunk_gen:

        ## [i, j]: Distance from shop img i to cons img j

        shop_to_cons_dists = torch.cdist(shop_img_embs, cons_img_embs[cons_img_idxs_chunk, :])

        ## [:, i]: Ordered closest shop img idxs to cons img i

        shop_to_cons_ordered_idxs = torch.argsort(shop_to_cons_dists, dim=0)

        ## [:, i]: ordered closest shop img item ids to cons img i 

        shop_to_cons_nearest_item_ids = shop_item_ids[shop_to_cons_ordered_idxs]

        ## [:, i]: True/False if, for each shop image, the cons img i is of the same item id

        shop_to_cons_hits = torch.eq(shop_to_cons_nearest_item_ids, cons_item_ids[cons_img_idxs_chunk])

        for k in k_values:

            k_corr = shop_img_embs.shape[0] if k == -1 else k

            ## [i]: Number of hits of cons img i (out of the k first)

            shop_to_cons_hits_sum = torch.sum(shop_to_cons_hits[:k_corr, ], dim=0)

            ## [i]: p/r_at_k of cons img i

            p_at_k = shop_to_cons_hits_sum / k_corr
            r_at_k = shop_to_cons_hits_sum / cons_shop_item_id_counts[cons_img_idxs_chunk]

            ## Accumulate results

            avg_p_at_k_dict[k] += torch.sum(p_at_k).item()
            avg_r_at_k_dict[k] += torch.sum(r_at_k).item()

    ## Average results
    
    for k in k_values:

        avg_p_at_k_dict[k] /= num_cons_imgs
        avg_r_at_k_dict[k] /= num_cons_imgs

    # Composite metrics

    avg_f1_at_k_dict = {k: 2 / ((1 / avg_p_at_k_dict[k]) + (1 / avg_r_at_k_dict[k])) for k in k_values}

    return avg_p_at_k_dict, avg_r_at_k_dict, avg_f1_at_k_dict


def compute_closest_idxs(
    shop_img_embs,
    shop_img_idxs,
    shop_item_ids,
    cons_img_embs,
    cons_img_idxs,
    cons_item_ids,
    desired_cons_img_idxs,
    num_desired_shop_imgs,
    cons_imgs_chunk_size,
    with_tqdm,
    device
):
    """
    """

    shop_img_idxs = torch.tensor(shop_img_idxs).to(device)
    shop_item_ids = shop_item_ids.to(device)
    
    # Result tensors

    num_desired_cons_imgs = len(desired_cons_img_idxs)

    ## [i, j]: j-th closest shop img idx to cons img with desired zidx i
    ## [i, j]: j-th closest shop item id to cons img with desired zidx i
    ## [i, j]: j-th closest shop dist to cons img with desired zidx i

    shop_to_desired_cons_ordered_closest_img_idxs = torch.empty(size=(num_desired_cons_imgs, num_desired_shop_imgs), dtype=int).to(device)
    shop_to_desired_cons_ordered_closest_item_ids = torch.empty(size=(num_desired_cons_imgs, num_desired_shop_imgs), dtype=int).to(device)
    shop_to_desired_cons_ordered_closest_dists = torch.empty(size=(num_desired_cons_imgs, num_desired_shop_imgs), dtype=float).to(device)
    
    # Counts of correct shop images

    ## [i]: Number of shop items with the same item id as cons img i

    cons_shop_item_id_counts = torch.empty_like(cons_item_ids)

    for cons_img_zidx, cons_item_id in enumerate(cons_item_ids):
        counts = torch.sum(torch.eq(shop_item_ids, cons_item_id))
        cons_shop_item_id_counts[cons_img_zidx] = counts

    # Preparing desired_cons_img_zidxs chunks

    desired_cons_img_zidxs = utils.arr.compute_zidxs(cons_img_idxs, desired_cons_img_idxs)
    desired_cons_img_zzidxs = np.arange(num_desired_cons_imgs)
    
    desired_cons_img_zidxs_chunk_gen = utils.chunk.chunk_partition_size(desired_cons_img_zidxs, cons_imgs_chunk_size)
    desired_cons_img_zzidxs_chunk_gen = utils.chunk.chunk_partition_size(desired_cons_img_zzidxs, cons_imgs_chunk_size)    
    
    chunk_idxs_gen = zip(desired_cons_img_zidxs_chunk_gen, desired_cons_img_zzidxs_chunk_gen)
    if with_tqdm: chunk_idxs_gen = tqdm(chunk_idxs_gen)

    for desired_cons_img_zidxs_chunk, desired_cons_img_zzidxs_chunk in chunk_idxs_gen:

        ## [j, i]: Distance from shop img zidx j to cons img with zidx i

        shop_to_cons_dists = torch.cdist(shop_img_embs, cons_img_embs[desired_cons_img_zidxs_chunk, :])

        ## [:, i]: Ordered closest shop img zidxs to cons img with zidx i

        shop_to_cons_ordered_closest_zidxs = torch.argsort(shop_to_cons_dists, dim=0).to(device)

        ## [:, i]: Ordered closest shop img idxs to cons img with zidx i 
        ## [:, i]: Ordered closest shop item ids to cons img with zidx i 
        ## [:, i]: Ordered closest shop img distances to cons img with zidx i 

        shop_to_cons_ordered_closest_img_idxs = shop_img_idxs[shop_to_cons_ordered_closest_zidxs]
        shop_to_cons_ordered_closest_item_ids = shop_item_ids[shop_to_cons_ordered_closest_zidxs]
        shop_to_cons_ordered_closest_dists = torch.take_along_dim(shop_to_cons_dists, shop_to_cons_ordered_closest_zidxs, dim=0)

        shop_to_desired_cons_ordered_closest_img_idxs[desired_cons_img_zzidxs_chunk, :] = torch.t(shop_to_cons_ordered_closest_img_idxs[:num_desired_shop_imgs, :])
        shop_to_desired_cons_ordered_closest_item_ids[desired_cons_img_zzidxs_chunk, :] = torch.t(shop_to_cons_ordered_closest_item_ids[:num_desired_shop_imgs, :])
        shop_to_desired_cons_ordered_closest_dists[desired_cons_img_zzidxs_chunk, :] = torch.t(shop_to_cons_ordered_closest_dists[:num_desired_shop_imgs, :]) 

    # Indexing counts of correct shop images

    cons_shop_item_id_counts = cons_shop_item_id_counts[desired_cons_img_zidxs]

    return (
        shop_to_desired_cons_ordered_closest_img_idxs,
        shop_to_desired_cons_ordered_closest_item_ids,
        shop_to_desired_cons_ordered_closest_dists,
        cons_shop_item_id_counts
    )



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
    parser.add_argument("eval_params_filename", help="filename of the evaluation params json file")
    parser.add_argument("exp_params_filename", help="filename of the experiment params json file")
    parser.add_argument("--terminal_silent", help="no terminal prints will be made", action="store_true")
    parser.add_argument("--no_tqdm", help="no tqdm bars will be shown", action="store_true")
    command_args = parser.parse_args()

    eval_params_filename = os.path.join(pathlib.Path.home(), "fashion_retrieval", command_args.eval_params_filename)
    exp_params_filename = os.path.join(pathlib.Path.home(), "fashion_retrieval", command_args.exp_params_filename)

    with_tqdm = not command_args.no_tqdm and not command_args.terminal_silent

    ####
    # EVALUATION PREREQUISITES
    ####


    # Read params

    eval_params = load_json_data(eval_params_filename)
    exp_params = load_json_data(exp_params_filename)

    # Experiment directory

    experiment_name__eval = eval_params["experiment_name"]
    experiment_name__exp = exp_params["experiment_name"]
    
    if experiment_name__eval != experiment_name__exp:
        raise ValueError("Experiment names do not match")


    ####
    # PREPARE LOGGER
    ####


    experiment_name = eval_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    log_filename = "eval_ctsrbm_examples__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)
    if os.path.exists(log_full_filename):
        os.remove(log_full_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = utils.log.Logger(logger_streams)
    
    logger.print("Command arguments:", " ".join(sys.argv))
    

    ####
    # PREPARE EVAL DATA
    ####


    eval_data = {}

    eval_data["script_name"] = sys.argv[0]
    eval_data["experiment_name"] = experiment_name
    eval_data["settings"] = {}
    eval_data["results"] = {}

    eval_data["settings"]["datasets"] = [
        "DeepFashion Consumer-to-shop Clothes Retrieval Benchmark"
    ]

    eval_data["settings"]["model_checkpoint"] = eval_params["settings"]["model_checkpoint"]

    eval_data["settings"]["command_args"] = vars(command_args)


    ####
    # GPU INITIALIZATION
    ####


    logger.print("Selecting CUDA devices")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_idx = eval_params["settings"]["device_idx"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in [device_idx]])

    device = torch.device(0)

    torch.cuda.empty_cache()

    eval_data["settings"]["gpu_usage"] = utils.mem.list_gpu_data([device_idx])
    eval_data["settings"]["hostname"] = socket.gethostname()

    logger.print("Selected CUDA devices")

    logger.print("Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=2))


    ####
    # DATA INITIALIZATION
    ####


    logger.print("Initializing image loader dataset")

    # Dataset initialization

    backbone_class = exp_params["settings"]["backbone"]["class"]

    backbone_image_transform = create_backbone_transform(backbone_class)
    backbone_image_transform.antialias = True

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")

    ctsrbm_dataset = deep_fashion_ctsrbm.ConsToShopClothRetrBmkImageLoader(ctsrbm_dataset_dir, img_transform=backbone_image_transform)

    logger.print("Initialized image loader dataset")


    ####
    # MODEL INITIALIZATION
    ####


    logger.print("Loading model from checkpoint")

    # Load components

    experiment_checkpoint_filename = eval_params["settings"]["model_checkpoint"]

    experiment_checkpoint_filename_full = os.path.join(
        experiment_dirname, experiment_checkpoint_filename
    )

    backbone, ret_head =\
    load_experiment_checkpoint(
        experiment_checkpoint_filename_full,
        exp_params,
        device
    )

    logger.print("Loaded model from checkpoint")


    # Build models

    ret_model = models.BackboneAndHead(backbone, ret_head).to(device)

    logger.print("Loaded model from checkpoint")


    ####
    # TRAIN PERFORMANCE METRICS
    ####


    logger.print("Train split examples begin")

    # Data loader initialization

    train_shop_img_idxs = ctsrbm_dataset.get_subset_indices(split="train", domain="shop")
    train_cons_img_idxs = ctsrbm_dataset.get_subset_indices(split="train", domain="consumer")

    train_shop_dataset = Subset(ctsrbm_dataset, train_shop_img_idxs)
    train_cons_dataset = Subset(ctsrbm_dataset, train_cons_img_idxs)

    batch_size = eval_params["settings"]["data_loading"]["batch_size"]
    num_workers = eval_params["settings"]["data_loading"]["num_workers"]

    train_shop_loader = DataLoader(
        train_shop_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    train_cons_loader = DataLoader(
        train_cons_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    # Embedding calculation

    logger.print("  Computing image embeddings")

    train_shop_img_embs, train_shop_item_ids = compute_embeddings_and_item_ids(
        ret_model,
        train_shop_loader,
        device,
        with_tqdm
    )
    
    train_cons_img_embs, train_cons_item_ids = compute_embeddings_and_item_ids(
        ret_model,
        train_cons_loader,
        device,
        with_tqdm
    )
    
    logger.print("  Computed image embeddings")

    # Example calculation

    logger.print("  Computing retrieval examples")

    train_desired_cons_img_idxs = eval_params["settings"]["train"]["desired_cons_img_idxs"]
    train_num_desired_shop_imgs = eval_params["settings"]["train"]["num_desired_shop_imgs"]
    cons_imgs_chunk_size = utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    (
        shop_to_desired_cons_ordered_closest_img_idxs,
        shop_to_desired_cons_ordered_closest_item_ids,
        shop_to_desired_cons_ordered_closest_dists,
        cons_shop_item_id_counts
    ) = compute_closest_idxs(
        train_shop_img_embs,
        train_shop_img_idxs,
        train_shop_item_ids,
        train_cons_img_embs,
        train_cons_img_idxs,
        train_cons_item_ids,
        train_desired_cons_img_idxs,
        train_num_desired_shop_imgs,
        cons_imgs_chunk_size,
        with_tqdm,
        device
    )

    logger.print("  Computed retrieval examples")

    # Save evaluation data

    eval_data["results"]["train"] = []

    train_desired_cons_img_zidxs = utils.arr.compute_zidxs(train_cons_img_idxs, train_desired_cons_img_idxs)
    for train_desired_cons_img_zzidx, train_desired_cons_img_zidx in enumerate(train_desired_cons_img_zidxs):

        cons_img_idx = train_cons_img_idxs[train_desired_cons_img_zidx]
        cons_item_id = train_cons_item_ids[train_desired_cons_img_zidx]
        shop_item_id_counts = cons_shop_item_id_counts[train_desired_cons_img_zzidx]
        closest_shop_img_idxs = shop_to_desired_cons_ordered_closest_img_idxs[train_desired_cons_img_zzidx, :]
        closest_shop_item_ids = shop_to_desired_cons_ordered_closest_item_ids[train_desired_cons_img_zzidx, :]
        closest_shop_img_dists = shop_to_desired_cons_ordered_closest_dists[train_desired_cons_img_zzidx, :]

        eval_data["results"]["train"].append({
            "cons_img_idx": int(cons_img_idx),
            "cons_item_id": cons_item_id.item(),
            "num_shop_imgs": shop_item_id_counts.item(),
            "closest_shop_img_idxs": closest_shop_img_idxs.tolist(),
            "closest_shop_item_ids": closest_shop_item_ids.tolist(),
            "closest_shop_img_dists": closest_shop_img_dists.tolist()
        })

    logger.print("  Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=4))

    logger.print("Train split examples end")

    
    ####
    # VAL PERFORMANCE METRICS
    ####


    logger.print("Validation split examples begin")

    # Data loader initialization

    val_shop_img_idxs = ctsrbm_dataset.get_subset_indices(split="val", domain="shop")
    val_cons_img_idxs = ctsrbm_dataset.get_subset_indices(split="val", domain="consumer")

    val_shop_dataset = Subset(ctsrbm_dataset, val_shop_img_idxs)
    val_cons_dataset = Subset(ctsrbm_dataset, val_cons_img_idxs)

    batch_size = eval_params["settings"]["data_loading"]["batch_size"]
    num_workers = eval_params["settings"]["data_loading"]["num_workers"]

    val_shop_loader = DataLoader(
        val_shop_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    val_cons_loader = DataLoader(
        val_cons_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    # Embedding calculation

    logger.print("  Computing image embeddings")

    val_shop_img_embs, val_shop_item_ids = compute_embeddings_and_item_ids(
        ret_model,
        val_shop_loader,
        device,
        with_tqdm
    )
    
    val_cons_img_embs, val_cons_item_ids = compute_embeddings_and_item_ids(
        ret_model,
        val_cons_loader,
        device,
        with_tqdm
    )
    
    logger.print("  Computed image embeddings")
    
    # Example calculation

    logger.print("  Computing retrieval examples")

    val_desired_cons_img_idxs = eval_params["settings"]["val"]["desired_cons_img_idxs"]
    val_num_desired_shop_imgs = eval_params["settings"]["val"]["num_desired_shop_imgs"]
    cons_imgs_chunk_size = utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    (
        shop_to_desired_cons_ordered_closest_img_idxs,
        shop_to_desired_cons_ordered_closest_item_ids,
        shop_to_desired_cons_ordered_closest_dists,
        cons_shop_item_id_counts
    ) = compute_closest_idxs(
        val_shop_img_embs,
        val_shop_img_idxs,
        val_shop_item_ids,
        val_cons_img_embs,
        val_cons_img_idxs,
        val_cons_item_ids,
        val_desired_cons_img_idxs,
        val_num_desired_shop_imgs,
        cons_imgs_chunk_size,
        with_tqdm,
        device
    )

    logger.print("  Computed retrieval examples")

    # Save evaluation data

    eval_data["results"]["val"] = []

    val_desired_cons_img_zidxs = utils.arr.compute_zidxs(val_cons_img_idxs, val_desired_cons_img_idxs)
    for val_desired_cons_img_zzidx, val_desired_cons_img_zidx in enumerate(val_desired_cons_img_zidxs):

        cons_img_idx = val_cons_img_idxs[val_desired_cons_img_zidx]
        cons_item_id = val_cons_item_ids[val_desired_cons_img_zidx]
        shop_item_id_counts = cons_shop_item_id_counts[val_desired_cons_img_zzidx]
        closest_shop_img_idxs = shop_to_desired_cons_ordered_closest_img_idxs[val_desired_cons_img_zzidx, :]
        closest_shop_item_ids = shop_to_desired_cons_ordered_closest_item_ids[val_desired_cons_img_zzidx, :]
        closest_shop_img_dists = shop_to_desired_cons_ordered_closest_dists[val_desired_cons_img_zzidx, :]

        eval_data["results"]["val"].append({
            "cons_img_idx": int(cons_img_idx),
            "cons_item_id": cons_item_id.item(),
            "num_shop_imgs": shop_item_id_counts.item(),
            "closest_shop_img_idxs": closest_shop_img_idxs.tolist(),
            "closest_shop_item_ids": closest_shop_item_ids.tolist(),
            "closest_shop_img_dists": closest_shop_img_dists.tolist()
        })

    logger.print("  Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=4))

    logger.print("Validation split examples end")


    ####
    # TEST PERFORMANCE METRICS
    ####


    logger.print("Test split examples begin")

    # Data loader initialization

    test_shop_img_idxs = ctsrbm_dataset.get_subset_indices(split="test", domain="shop")
    test_cons_img_idxs = ctsrbm_dataset.get_subset_indices(split="test", domain="consumer")

    test_shop_dataset = Subset(ctsrbm_dataset, test_shop_img_idxs)
    test_cons_dataset = Subset(ctsrbm_dataset, test_cons_img_idxs)

    batch_size = eval_params["settings"]["data_loading"]["batch_size"]
    num_workers = eval_params["settings"]["data_loading"]["num_workers"]

    test_shop_loader = DataLoader(
        test_shop_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    test_cons_loader = DataLoader(
        test_cons_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    # Embedding calculation

    logger.print("  Computing image embeddings")

    test_shop_img_embs, test_shop_item_ids = compute_embeddings_and_item_ids(
        ret_model,
        test_shop_loader,
        device,
        with_tqdm
    )
    
    test_cons_img_embs, test_cons_item_ids = compute_embeddings_and_item_ids(
        ret_model,
        test_cons_loader,
        device,
        with_tqdm
    )
    
    logger.print("  Computed image embeddings")
    
    # Example calculation

    logger.print("  Computing retrieval examples")

    test_desired_cons_img_idxs = eval_params["settings"]["test"]["desired_cons_img_idxs"]
    test_num_desired_shop_imgs = eval_params["settings"]["test"]["num_desired_shop_imgs"]
    cons_imgs_chunk_size = utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    (
        shop_to_desired_cons_ordered_closest_img_idxs,
        shop_to_desired_cons_ordered_closest_item_ids,
        shop_to_desired_cons_ordered_closest_dists,
        cons_shop_item_id_counts
    ) = compute_closest_idxs(
        test_shop_img_embs,
        test_shop_img_idxs,
        test_shop_item_ids,
        test_cons_img_embs,
        test_cons_img_idxs,
        test_cons_item_ids,
        test_desired_cons_img_idxs,
        test_num_desired_shop_imgs,
        cons_imgs_chunk_size,
        with_tqdm,
        device
    )

    logger.print("  Computed retrieval examples")

    # Save evaluation data

    eval_data["results"]["test"] = []

    test_desired_cons_img_zidxs = utils.arr.compute_zidxs(test_cons_img_idxs, test_desired_cons_img_idxs)
    for test_desired_cons_img_zzidx, test_desired_cons_img_zidx in enumerate(test_desired_cons_img_zidxs):

        cons_img_idx = test_cons_img_idxs[test_desired_cons_img_zidx]
        cons_item_id = test_cons_item_ids[test_desired_cons_img_zidx]
        shop_item_id_counts = cons_shop_item_id_counts[test_desired_cons_img_zzidx]
        closest_shop_img_idxs = shop_to_desired_cons_ordered_closest_img_idxs[test_desired_cons_img_zzidx, :]
        closest_shop_item_ids = shop_to_desired_cons_ordered_closest_item_ids[test_desired_cons_img_zzidx, :]
        closest_shop_img_dists = shop_to_desired_cons_ordered_closest_dists[test_desired_cons_img_zzidx, :]

        eval_data["results"]["test"].append({
            "cons_img_idx": int(cons_img_idx),
            "cons_item_id": cons_item_id.item(),
            "num_shop_imgs": shop_item_id_counts.item(),
            "closest_shop_img_idxs": closest_shop_img_idxs.tolist(),
            "closest_shop_item_ids": closest_shop_item_ids.tolist(),
            "closest_shop_img_dists": closest_shop_img_dists.tolist()
        })

    logger.print("  Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=4))

    logger.print("Test split examples end")
    

    ####
    # SAVE RESULTS
    ####


    eval_data_filename = os.path.join(
        experiment_dirname, "eval_ctsrbm_examples__data.json"
    )

    logger.print("Saving results to \"{:s}\"".format(
        eval_data_filename
    ))

    save_json_data(eval_data_filename, eval_data)

    logger.print("Saved results to \"{:s}\"".format(
        eval_data_filename
    ))
