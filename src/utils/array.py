import numpy as np



def compute_zidxs(arr, vals):
    """
    Computes the indices of the first occurrence of values in an original array.

    :param arr: np.ndarray
        1D original array to search in.
    :param vals: np.ndarray
        1D original array with the values to search.

    :return: np.ndarray
        The computed first occurrence indices.
        The following satisfies: `arr[zidxs] = vals`.
    """

    zidxs = []
    for val in vals:
        argw = np.argwhere(arr == val)
        if argw.shape[0] == 0:
            raise ValueError("Value {:s} not found in arr".format(str(val)))
        zidxs.append(argw[0][0])

    return np.asarray(zidxs)
