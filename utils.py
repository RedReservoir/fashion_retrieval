import sys
import os
import uuid

import torch
import torchvision.transforms.functional


def get_num_bytes(obj):

    num_bytes = sys.getsizeof(obj)

    if type(obj) is list:
        for el in obj:
            num_bytes += get_num_bytes(el)
    if type(obj) is dict:
        for key, val in obj.items():
            num_bytes += get_num_bytes(key)
            num_bytes += get_num_bytes(val)

    return num_bytes


def sprint_fancy_num_bytes(num_bytes):
    
    units = ["B", "KiB", "MiB", "GiB"]
    unit_idx = 0

    while num_bytes > 1024:
        num_bytes /= 1024
        unit_idx += 1

    byte_fancy_str = "{:.3f}{:s}".format(num_bytes, units[unit_idx])

    return byte_fancy_str


def sprint_fancy_time_diff(time_diff):
    """
    Generates a fancy string representation of a time difference.

    :param time_diff: float
        Time difference, in seconds.

    :return: str
        Fancy string representation of the time difference.
    """

    minutes = int(time_diff // 60)
    time_diff %= 60
    seconds = int(time_diff // 1)
    time_diff %= 1
    milliseconds = int(time_diff * 1000 // 1)

    return "{:02d}:{:02d}.{:03d}".format(
        minutes, seconds, milliseconds
    )


def generate_unused_filename(dir="."):
    """
    Finds a random unused filename.

    :param dir: str
        Target directory in which to check for files.
    
    :return: str
        A filename not currently used.
    """

    filename = os.path.join(dir, str(uuid.uuid4()))
    while os.path.exists(filename):
        filename = os.path.join(dir, str(uuid.uuid4()))
    
    return filename


def cutdown_list(my_list, ratio):
    """
    TODO
    """

    new_len = round(len(my_list) * ratio)
    return my_list[:new_len]



class EarlyStopper:
    

    def __init__(self, patience=1, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float("inf")


    def early_stop(self, val_loss):
        
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False