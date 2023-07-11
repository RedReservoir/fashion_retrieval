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
    
    units = ["B", "KB", "MB", "GB"]
    unit_idx = 0

    while num_bytes > 1024:
        num_bytes /= 1024
        unit_idx += 1

    byte_fancy_str = "{:.2f}{:s}".format(num_bytes, units[unit_idx])

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
