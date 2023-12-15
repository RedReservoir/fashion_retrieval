from pynvml import *
nvmlInit()

import torch
import sys

import src.utils.memory



def sprint_memory_usage(idxs, num_spaces=0, file=sys.stdout):
    """
    Generates a string showing GPU memoru usage.
    Implemented with PyTorch and cuda.

    :param idxs: list
        Indices of cuda devices, according to nvidia-smi.
    :param num_spaces: int, default=0
        Number of spaces with which to tabulate lines.

    :return: str
        String showing GPU memoru usage.
    """
    
    msgs = []

    for zidx, idx in enumerate(idxs):

        h = nvmlDeviceGetHandleByIndex(idx)
        info = nvmlDeviceGetMemoryInfo(h)
        
        gpu_name = torch.cuda.get_device_name(zidx)
        
        total_mem_str = src.utils.memory.sprint_fancy_num_bytes(info.total)
        used_mem_str = src.utils.memory.sprint_fancy_num_bytes(info.used)
        use_perc = info.used / info.total * 100

        msgs.append((" " * num_spaces) + "Device ID {:2d}: {:s} / {:s} ({:6.2f}%) - {:s}".format(
            idx,
            used_mem_str,
            total_mem_str,
            use_perc,
            gpu_name
        ))

    final_msg = "\n".join(msgs)
    return final_msg



def list_gpu_data(idxs):
    """
    Generates a list of strings showing GPU names and their size.
    Implemented with PyTorch and cuda.

    :param idxs: list
        Indices of cuda devices, according to nvidia-smi.

    :return: list
        List of dicts with GPU names and their sizes.
    """
    
    gpu_data_list = []

    for zidx, idx in enumerate(idxs):

        h = nvmlDeviceGetHandleByIndex(idx)
        info = nvmlDeviceGetMemoryInfo(h)
        
        gpu_data = {
            "device_id": idx,
            "device_name": torch.cuda.get_device_name(zidx),
            "device_size": info.total,
            "device_size_fancy": src.utils.memory.sprint_fancy_num_bytes(info.total)
        }

        gpu_data_list.append(gpu_data)

    return gpu_data_list



def list_gpu_usage(idxs):
    """
    Generates a list of strings showing GPU names, sizes and usages.
    Implemented with PyTorch and cuda.

    :param idxs: list
        Indices of cuda devices, according to nvidia-smi.

    :return: list
        List of dicts with GPU names, their sizes and usages.
    """
    
    gpu_data_list = []

    for zidx, idx in enumerate(idxs):

        h = nvmlDeviceGetHandleByIndex(idx)
        info = nvmlDeviceGetMemoryInfo(h)
        
        gpu_data = {
            "device_id": idx,
            "device_name": torch.cuda.get_device_name(zidx),
            "device_size": info.total,
            "device_size_fancy": src.utils.memory.sprint_fancy_num_bytes(info.total),
            "device_usage": info.used,
            "device_usage_fancy": src.utils.memory.sprint_fancy_num_bytes(info.used)
        }

        gpu_data_list.append(gpu_data)

    return gpu_data_list
