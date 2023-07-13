from pynvml import *
nvmlInit()



def sprint_fancy_num_bytes(num_bytes):
    """
    Generates a fancy string representation of memory quantity.

    :param num_bytes: float
        Number of bytes.

    :return: str
        Fancy string representation of the memory quantity.
    """
    
    units = ["B", "KiB", "MiB", "GiB"]
    unit_idx = 0

    while num_bytes > 1024:
        num_bytes /= 1024
        unit_idx += 1

    byte_fancy_str = "{:7.3f} {:3s}".format(num_bytes, units[unit_idx])

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

    return num_bytes


def sprint_memory_usage(idxs, num_spaces=0):
    """
    Generates a string showing GPU memoru usage with PyTorch and cuda.

    :param idxs: list
        Indices of cuda devices, according to nvidia-smi.
    :param num_spaces: int, default=0
        Number of spaces with which to tabulate lines.

    :return: str
        String showing GPU memoru usage.
    """
    
    msgs = []

    for idx in idxs:

        h = nvmlDeviceGetHandleByIndex(idx)
        info = nvmlDeviceGetMemoryInfo(h)
        
        total_mem_str = sprint_fancy_num_bytes(info.total)
        used_mem_str = sprint_fancy_num_bytes(info.used)
        use_perc = info.used / info.total * 100

        msgs.append((" " * num_spaces) + "Device ID {:2d}: {:s} / {:s} ({:6.2f}%)".format(
            idx,
            used_mem_str,
            total_mem_str,
            use_perc
        ))

    final_msg = "\n".join(msgs)
    return final_msg
