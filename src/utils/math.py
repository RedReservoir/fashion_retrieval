import numpy as np



def get_closest_perc_div(x, f):
    """
    Returns the divisor of x closest to f.

    :param x: int
    :param p: float
    
    :return: int
        The closest divisor.
    """

    if f == float("inf"):
        return x

    div_arr = np.asarray([i for i in range(1, x+1) if x % i == 0])
    dist_arr = np.abs(div_arr - f)
    div_idx = np.argmin(dist_arr)
    
    return div_arr[div_idx]
