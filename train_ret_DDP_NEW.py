import os
import shutil
import sys
import pathlib
import json
import argparse
import json
import socket

import numpy as np
import pandas as pd

from time import time
from datetime import datetime
from itertools import chain
from functools import reduce

from tqdm import tqdm

import utils

from datasets import deep_fashion
from arch import backbones, models, heads

import torch
import torchvision

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



########
# COMPONENT FUNCTIONS
########



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



########
# TRAIN & EVAL FUNCTIONS
########



########
# MAIN FUNCTION
########



def main(
    rank,
    world_size,
    command_args,
    experiment_params,
    experiment_data
):
    
    experiment_name = experiment_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)

    ####
    # PREPARE LOGGER
    ####

    log_filename = "experiment_logs.txt"
    log_full_filename = os.path.join(experiment_dirname, log_filename)

    logger_streams = [log_full_filename]
    if not command_args.silent: logger_streams.append(sys.stdout)

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
# MAIN SCRIPT
########



if __name__ == "__main__":

    datetime_now_str = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")

    ####
    # COMMAND ARGUMENTS
    ####

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_params_filename", help="filename of the experiment params json file inside the \"exp_params\" directory")
    parser.add_argument("--silent", help="no terminal prints will be made", action="store_true")
    parser.add_argument("--notqdm", help="no tqdm bars will be shown", action="store_true")
    parser.add_argument("--reset", help="experiment directory will be reset", action="store_true")
    command_args = parser.parse_args()

    ####
    # EXPERIMENT PREREQUISITES
    ####

    # Read experiment params

    experiment_params_filename = os.path.join(pathlib.Path.home(), "fashion_retrieval", "exp_params", command_args.experiment_params_filename)

    experiment_params_file = open(experiment_params_filename, 'r')
    experiment_params = json.load(experiment_params_file)
    experiment_params_file.close()

    # Check and create experiment directory

    experiment_name = experiment_params["experiment_name"]
    experiment_dirname = os.path.join(pathlib.Path.home(), "data", "fashion_retrieval", experiment_name)
    
    if os.path.exists(experiment_dirname) and command_args.reset:
        shutil.rmtree(experiment_dirname)

    if not os.path.exists(experiment_dirname):
        os.mkdir(experiment_dirname)

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
    
    experiment_data = load_experiment_data(experiment_data_filename)

    ####
    # MULTIPROCESSING START
    ####

    world_size = len(experiment_params["settings"]["device_idxs"])
    
    torch.multiprocessing.spawn(
        main,
        args=[
            world_size,
            command_args,
            experiment_params,
            experiment_data
        ],
        nprocs=world_size
    )
