import os
import sys
import pathlib
import argparse

from tqdm import tqdm

from datetime import datetime

import socket

########

import numpy as np

import torch

from torch.utils.data import DataLoader, Subset

########

from src.datasets import deep_fashion_ctsrbm

import src.utils.train
import src.utils.log
import src.utils.dict
import src.utils.list
import src.utils.nvgpu
import src.utils.signal
import src.utils.time
import src.utils.chunk
import src.utils.comps
import src.utils.json
import src.utils.dgi



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


def compute_retrieval_accuracy(
    shop_img_embs,
    shop_item_ids,
    cons_img_embs,
    cons_item_ids,
    cons_imgs_chunk_size,
    with_tqdm,
    device
):

    num_cons_imgs = cons_img_embs.shape[0]
    cons_img_accs = torch.empty((num_cons_imgs), dtype=float).to(device)

    # Precision and recall metrics

    ## [i]: Number of shop items with the same item id as cons img i

    cons_shop_item_id_counts = torch.empty_like(cons_item_ids)

    for idx, cons_item_id in enumerate(cons_item_ids):
        counts = torch.sum(torch.eq(shop_item_ids, cons_item_id))
        cons_shop_item_id_counts[idx] = counts

    # Computing metrics in chunks

    cons_img_zidxs_chunk_gen = src.utils.chunk.chunk_partition_size(np.arange(num_cons_imgs), cons_imgs_chunk_size)
    if with_tqdm: cons_img_zidxs_chunk_gen = tqdm(cons_img_zidxs_chunk_gen)

    for cons_img_zidxs_chunk in cons_img_zidxs_chunk_gen:

        ## [i, j]: Distance from shop img i to cons img j

        shop_to_cons_dists = torch.cdist(shop_img_embs, cons_img_embs[cons_img_zidxs_chunk, :])

        ## [:, i]: Ordered closest shop img idxs to cons img i

        shop_to_cons_ordered_closest_zidxs = torch.argsort(shop_to_cons_dists, dim=0)

        ## [:, i]: ordered closest shop img item ids to cons img i 

        shop_to_cons_ordered_closest_item_ids = shop_item_ids[shop_to_cons_ordered_closest_zidxs]

        ## [:, i]: True/False if, for each shop image, the cons img i is of the same item id

        shop_to_cons_hits = torch.eq(shop_to_cons_ordered_closest_item_ids, cons_item_ids[cons_img_zidxs_chunk])

        # Accuracy calculation

        cons_shop_item_id_counts_chunk = cons_shop_item_id_counts[cons_img_zidxs_chunk]

        ## [i]: Number of hits of cons img i (out of the k first)

        shop_to_cons_hits_sum = torch.zeros((shop_to_cons_hits.shape[1]), dtype=int).to(device)

        for cons_img_zidx in range(shop_to_cons_hits.shape[1]):

            num_shop_items = cons_shop_item_id_counts_chunk[cons_img_zidx]
            shop_to_cons_hits_sum[cons_img_zidx] = torch.sum(shop_to_cons_hits[:num_shop_items, cons_img_zidx]).item()

        ## [i]: Retrieval accuracy of cons img i

        acc = shop_to_cons_hits_sum / cons_shop_item_id_counts_chunk

        ## Accumulate results

        cons_img_accs[cons_img_zidxs_chunk] = acc.double()

    return cons_img_accs, cons_shop_item_id_counts



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

    with_tqdm = not command_args.no_tqdm
    

    ####
    # EVALUATION PREREQUISITES
    ####


    # Read params

    eval_params = src.utils.json.load_json_dict(eval_params_filename)
    exp_params = src.utils.json.load_json_dict(exp_params_filename)

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

    log_filename = "eval_ctsrbm_accuracy__logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)
    if os.path.exists(log_full_filename):
        os.remove(log_full_filename)

    logger_streams = [log_full_filename]
    if not command_args.terminal_silent: logger_streams.append(sys.stdout)

    logger = src.utils.log.Logger(logger_streams)

    sys.stderr = logger
    
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

    eval_data["settings"]["gpu_usage"] = src.utils.nvgpu.list_gpu_data([device_idx])
    eval_data["settings"]["hostname"] = socket.gethostname()

    logger.print("Selected CUDA devices")

    logger.print("Current memory usage:")
    logger.print(src.utils.nvgpu.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=2))


    ####
    # MODEL INITIALIZATION
    ####

    logger.print("Loading model from checkpoint")

    # Load components

    backbone = src.utils.comps.create_backbone(exp_params["settings"]["backbone"])
    backbone = backbone.to(device)

    ret_head = src.utils.comps.create_head(
        backbone,
        exp_params["settings"]["head"]
    )
    ret_head = ret_head.to(device)


    experiment_checkpoint_filename = os.path.join(
        experiment_dirname, eval_params["settings"]["model_checkpoint"]
    )

    experiment_checkpoint = torch.load(experiment_checkpoint_filename)

    backbone.load_state_dict(experiment_checkpoint["backbone_state_dict"])
    ret_head.load_state_dict(experiment_checkpoint["ret_head_state_dict"])

    # Build models

    ret_model = torch.nn.Sequential(backbone, ret_head).to(device)

    logger.print("Loaded model from checkpoint")
    

    ####
    # DATA INITIALIZATION
    ####


    logger.print("Initializing image loader dataset")

    # Dataset initialization

    ctsrbm_dataset_dir = os.path.join(pathlib.Path.home(), "data", "DeepFashion", "Consumer-to-shop Clothes Retrieval Benchmark")
    backbone_image_transform = backbone.get_image_transform()

    ctsrbm_dataset = deep_fashion_ctsrbm.ConsToShopClothRetrBmkImageLoader(ctsrbm_dataset_dir, img_transform=backbone_image_transform)

    logger.print("Initialized image loader dataset")


    ####
    # TRAIN PERFORMANCE METRICS
    ####


    logger.print("Train split accuracy begin")

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

    # Retrieval accuracy

    logger.print("  Computing retrieval accuracy")

    cons_imgs_chunk_size = src.utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    cons_img_accs, cons_shop_item_id_counts =\
        compute_retrieval_accuracy(
            train_shop_img_embs,
            train_shop_item_ids,
            train_cons_img_embs,
            train_cons_item_ids,
            cons_imgs_chunk_size,
            with_tqdm,
            device
        )

    logger.print("  Computed accuracy accuracy")

    # Save evaluation data

    eval_data["results"]["train"] = [
        (img_idx.item(), retr_acc.item(), num_shop_imgs.item())
        for (img_idx, retr_acc, num_shop_imgs)
        in zip(train_cons_idxs, cons_img_accs, cons_shop_item_id_counts)
    ]

    logger.print("  Current memory usage:")
    logger.print(src.utils.nvgpu.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=4))

    logger.print("Train split accuracy end")


    ####
    # VAL PERFORMANCE METRICS
    ####


    logger.print("Validation split accuracy begin")

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
    
    # Retrieval accuracy

    logger.print("  Computing retrieval accuracy")

    cons_imgs_chunk_size = src.utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    cons_img_accs, cons_shop_item_id_counts =\
        compute_retrieval_accuracy(
            val_shop_img_embs,
            val_shop_item_ids,
            val_cons_img_embs,
            val_cons_item_ids,
            cons_imgs_chunk_size,
            with_tqdm,
            device
        )

    logger.print("  Computed retrieval accuracy")

    # Save evaluation data

    # Save evaluation data

    eval_data["results"]["val"] = [
        (img_idx.item(), retr_acc.item(), num_shop_imgs.item())
        for (img_idx, retr_acc, num_shop_imgs)
        in zip(val_cons_idxs, cons_img_accs, cons_shop_item_id_counts)
    ]

    logger.print("Current memory usage:")
    logger.print(src.utils.nvgpu.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=2))

    logger.print("Validation split accuracy end")


    ####
    # TEST PERFORMANCE METRICS
    ####


    logger.print("Test split accuracy begin")

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

    # Retrieval accuracy

    logger.print("  Computing retrieval accuracy")

    cons_imgs_chunk_size = src.utils.dict.chain_get(
        eval_params,
        "settings", "cons_imgs_chunk_size",
        default=1000
    )

    cons_img_accs, cons_shop_item_id_counts =\
        compute_retrieval_accuracy(
            test_shop_img_embs,
            test_shop_item_ids,
            test_cons_img_embs,
            test_cons_item_ids,
            cons_imgs_chunk_size,
            with_tqdm,
            device
        )
    
    logger.print("  Computed retrieval accuracy")

    # Save evaluation data

    eval_data["results"]["test"] = [
        (img_idx.item(), retr_acc.item(), num_shop_imgs.item())
        for (img_idx, retr_acc, num_shop_imgs)
        in zip(test_cons_idxs, cons_img_accs, cons_shop_item_id_counts)
    ]

    logger.print("Current memory usage:")
    logger.print(src.utils.nvgpu.sprint_memory_usage([eval_params["settings"]["device_idx"]], num_spaces=2))

    logger.print("Test split accuracy end")


    ####
    # SAVE RESULTS
    ####


    eval_data_filename = os.path.join(
        experiment_dirname, "eval_ctsrbm_accuracy__data.json"
    )

    logger.print("Saving results to \"{:s}\"".format(
        eval_data_filename
    ))

    src.utils.json.save_json_dict(
        eval_data,
        eval_data_filename
        )

    logger.print("Saved results to \"{:s}\"".format(
        eval_data_filename
    ))
