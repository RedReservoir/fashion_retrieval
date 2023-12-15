import torch
import sys



def sprint_fancy_num_bytes(num_bytes, unit=None):
    """
    Generates a fancy string representation of memory quantity.

    :param num_bytes: float
        Number of bytes.
    :param unit: str
        Desired unit


    :return: str
        Fancy string representation of the memory quantity.
    """
    
    units = ["B", "KiB", "MiB", "GiB"]
    unit_idx = 0

    while num_bytes > 1024 or ((unit is not None) and (unit != units[unit_idx])):
        num_bytes /= 1024
        unit_idx += 1

    byte_fancy_str = "{:.2f} {:3s}".format(num_bytes, units[unit_idx])

    return byte_fancy_str


def get_num_bytes(obj):
    """
    Computes the number of bytes that an object weights.
    Supports basic data types, lists and dicts.

    :param obj: any
        Object to compute memory from.

    :return: float
        Number of bytes that the object weights.
    """
    
    num_bytes = sys.getsizeof(obj)

    if type(obj) is list:
        for el in obj:
            num_bytes += get_num_bytes(el)
    if type(obj) is dict:
        for key, val in obj.items():
            num_bytes += get_num_bytes(key)
            num_bytes += get_num_bytes(val)
    if type(obj) is torch.Tensor:
        num_bytes += obj.nelement() * obj.element_size()

    return num_bytes
