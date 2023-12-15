import numpy as np
import math



def chunk_partition_size(arr, chunk_size):
    """
    Partitions a numpy array into sub-arrays.

    :param arr: np.ndarray
        1D numpy array to partition.
    :param chunk_size:
        Desired size of the sub-arrays.

    :return: list of np.ndarray
        List containing the sub-arrays.
    """

    return np.array_split(arr, max(1, math.ceil(arr.shape[0] / chunk_size)))



def chunk_partition_num(arr, num_chunks):
    """
    Partitions a numpy array into sub-arrays.

    :param arr: np.ndarray
        1D numpy array to partition.
    :param num_chunks:
        Desired number of chunks.

    :return: list of np.ndarray
        List containing the sub-arrays.
    """

    return np.array_split(arr, num_chunks)