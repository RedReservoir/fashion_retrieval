import numpy as np


def compute_zidxs(arr, vals):
    """
    TODO
    """

    zidxs = []
    for val in vals:
        argw = np.argwhere(arr == val)
        if argw.shape[0] == 0:
            raise ValueError("Value {:s} not found in arr".format(str(val)))
        zidxs.append(argw[0][0])

    return np.asarray(zidxs)
