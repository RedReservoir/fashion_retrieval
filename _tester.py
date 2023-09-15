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
import utils.log_new
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


#


if __name__ == "__main__":

    log_filename = "test_logs.txt"
    logger_streams = [
        log_filename,
        sys.stdout
    ]

    logger = utils.log_new.Logger(logger_streams)

    sys.stderr = logger

    logger.print("HOLA")
    logger.print("division: {:d}".format(6 // 3))
    logger.print("division: {:d}".format(6 // 0))