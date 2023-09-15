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
    with_tqdm,
    device
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
    with_tqdm,
    device
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

    # Computing metrics in chunks

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

        # Metric calculation for each k value

        cons_shop_item_id_counts_chunk = cons_shop_item_id_counts[cons_img_idxs_chunk]

        for k in k_values:

            if k == "same":

                ## [i]: Number of hits of cons img i (out of the k first)

                shop_to_cons_hits_sum = torch.zeros((shop_to_cons_hits.shape[1]), dtype=int).to(device)

                for cons_img_zidx in range(shop_to_cons_hits.shape[1]):

                    num_shop_items = cons_shop_item_id_counts_chunk[cons_img_zidx]
                    shop_to_cons_hits_sum[cons_img_zidx] = torch.sum(shop_to_cons_hits[:num_shop_items, cons_img_zidx]).item()

                ## [i]: Retrieval accuracy of cons img i

                acc = shop_to_cons_hits_sum / cons_shop_item_id_counts_chunk

                ## Accumulate results

                avg_p_at_k_dict[k] += torch.sum(acc).item()
                avg_r_at_k_dict[k] += torch.sum(acc).item()

            else:

                if k == "all":
                    k_corr =  shop_img_embs.shape[0]
                else:
                    k_corr = k

                ## [i]: Number of hits of cons img i (out of the k first)

                shop_to_cons_hits_sum = torch.sum(shop_to_cons_hits[:k_corr, :], dim=0)

                ## [i]: p/r_at_k of cons img i

                p_at_k = shop_to_cons_hits_sum / k_corr
                r_at_k = shop_to_cons_hits_sum / cons_shop_item_id_counts_chunk

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

    log_filename = "eval_ctsrbm_metrics__logs.txt"
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


    logger.print("Train split metrics begin")

    # Data loader initialization

    train_shop_idxs = ctsrbm_dataset.get_subset_indices(split="train", domain="shop")
    train_cons_idxs = ctsrbm_dataset.get_subset_indices(split="train", domain="consumer")

    train_shop_dataset = Subset(ctsrbm_dataset, train_shop_idxs)
    train_cons_dataset = Subset(ctsrbm_dataset, train_cons_idxs)

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

    train_shop_img_embs, train_shop_item_ids =\
        compute_embeddings_and_item_ids(
            ret_model,
            train_shop_loader,
            with_tqdm,
            device
        )
    
    train_cons_img_embs, train_cons_item_ids =\
        compute_embeddings_and_item_ids(
            ret_model,
            train_cons_loader,
            with_tqdm,
            device
        )
    
    logger.print("  Computed image embeddings")

    # Metric calculation

    logger.print("  Computing performance metrics")

    k_values = eval_params["settings"]["k_values"]
    cons_imgs_chunk_size = utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    avg_p_at_k_dict, avg_r_at_k_dict, avg_f1_at_k_dict =\
        compute_performance_metrics(
            train_shop_img_embs,
            train_shop_item_ids,
            train_cons_img_embs,
            train_cons_item_ids,
            k_values,
            cons_imgs_chunk_size,
            with_tqdm,
            device
        )

    logger.print("  Computed performance metrics")

    # Save evaluation data

    eval_data["results"]["train"] = {
        "avg_p_at_k_dict": avg_p_at_k_dict,
        "avg_r_at_k_dict": avg_r_at_k_dict,
        "avg_f1_at_k_dict": avg_f1_at_k_dict
    }

    logger.print("  Results:")
    logger.print("  Avg prec@k:")
    logger.print("  ", avg_p_at_k_dict, sep="")
    logger.print("  Avg rec@k:")
    logger.print("  ", avg_r_at_k_dict, sep="")
    logger.print("  Avg f1@k:")
    logger.print("  ", avg_f1_at_k_dict, sep="")

    logger.print("  Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=4))

    logger.print("Train split metrics end")


    ####
    # VAL PERFORMANCE METRICS
    ####


    logger.print("Validation split metrics begin")

    # Data loader initialization

    val_shop_idxs = ctsrbm_dataset.get_subset_indices(split="val", domain="shop")
    val_cons_idxs = ctsrbm_dataset.get_subset_indices(split="val", domain="consumer")

    val_shop_dataset = Subset(ctsrbm_dataset, val_shop_idxs)
    val_cons_dataset = Subset(ctsrbm_dataset, val_cons_idxs)

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

    val_shop_img_embs, val_shop_item_ids =\
        compute_embeddings_and_item_ids(
            ret_model,
            val_shop_loader,
            with_tqdm,
            device
        )
    
    val_cons_img_embs, val_cons_item_ids =\
        compute_embeddings_and_item_ids(
            ret_model,
            val_cons_loader,
            with_tqdm,
            device
        )
    
    logger.print("  Computed image embeddings")
    
    # Metric calculation

    logger.print("  Computing performance metrics")

    k_values = eval_params["settings"]["k_values"]
    cons_imgs_chunk_size = utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    avg_p_at_k_dict, avg_r_at_k_dict, avg_f1_at_k_dict =\
        compute_performance_metrics(
            val_shop_img_embs,
            val_shop_item_ids,
            val_cons_img_embs,
            val_cons_item_ids,
            k_values,
            cons_imgs_chunk_size,
            with_tqdm,
            device
        )

    logger.print("  Computed performance metrics")

    # Save evaluation data

    eval_data["results"]["val"] = {
        "avg_p_at_k_dict": avg_p_at_k_dict,
        "avg_r_at_k_dict": avg_r_at_k_dict,
        "avg_f1_at_k_dict": avg_f1_at_k_dict
    }
    
    logger.print("  Results:")
    logger.print("  Avg prec@k:")
    logger.print("  ", avg_p_at_k_dict, sep="")
    logger.print("  Avg rec@k:")
    logger.print("  ", avg_r_at_k_dict, sep="")
    logger.print("  Avg f1@k:")
    logger.print("  ", avg_f1_at_k_dict, sep="")

    logger.print("Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=2))

    logger.print("Validation split metrics end")


    ####
    # TEST PERFORMANCE METRICS
    ####


    logger.print("Test split metrics begin")

    # Data loader initialization

    test_shop_idxs = ctsrbm_dataset.get_subset_indices(split="test", domain="shop")
    test_cons_idxs = ctsrbm_dataset.get_subset_indices(split="test", domain="consumer")

    test_shop_dataset = Subset(ctsrbm_dataset, test_shop_idxs)
    test_cons_dataset = Subset(ctsrbm_dataset, test_cons_idxs)

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

    test_shop_img_embs, test_shop_item_ids =\
        compute_embeddings_and_item_ids(
            ret_model,
            test_shop_loader,
            with_tqdm,
            device
        )
    
    test_cons_img_embs, test_cons_item_ids =\
        compute_embeddings_and_item_ids(
            ret_model,
            test_cons_loader,
            with_tqdm,
            device
        )
    
    logger.print("  Computed image embeddings")

    # Metric calculation

    logger.print("  Computing performance metrics")

    k_values = eval_params["settings"]["k_values"]
    cons_imgs_chunk_size = utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    avg_p_at_k_dict, avg_r_at_k_dict, avg_f1_at_k_dict =\
        compute_performance_metrics(
            test_shop_img_embs,
            test_shop_item_ids,
            test_cons_img_embs,
            test_cons_item_ids,
            k_values,
            cons_imgs_chunk_size,
            with_tqdm,
            device
        )
    
    logger.print("  Computed performance metrics")

    # Save evaluation data

    eval_data["results"]["test"] = {
        "avg_p_at_k_dict": avg_p_at_k_dict,
        "avg_r_at_k_dict": avg_r_at_k_dict,
        "avg_f1_at_k_dict": avg_f1_at_k_dict
    }
    
    logger.print("  Results:")
    logger.print("  Avg prec@k:")
    logger.print("  ", avg_p_at_k_dict, sep="")
    logger.print("  Avg rec@k:")
    logger.print("  ", avg_r_at_k_dict, sep="")
    logger.print("  Avg f1@k:")
    logger.print("  ", avg_f1_at_k_dict, sep="")

    logger.print("Current memory usage:")
    logger.print(utils.mem.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=2))

    logger.print("Test split metrics end")


    ####
    # SAVE RESULTS
    ####


    eval_data_filename = os.path.join(
        experiment_dirname, "eval_ctsrbm_metrics__data.json"
    )

    logger.print("Saving results to \"{:s}\"".format(
        eval_data_filename
    ))

    save_json_data(eval_data_filename, eval_data)

    logger.print("Saved results to \"{:s}\"".format(
        eval_data_filename
    ))
