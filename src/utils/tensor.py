import sys
import torch

import src.utils.memory



def print_tensor_info(tensor, name=None, nl=False, file=sys.stdout):
    """
    Prints tensor information.
    Used for debug purposes.

    :param tensor: torch.Tensor
        Tensor to print information from.
    :param name: str, optional
        Name of the tensor variable.
        If not provided, name will not be shown.
    :param nl: bool, default=False
        Whether to print a newline after the tensor info.
    :param file: io.TextIO, default=sys.stdout
        Output stream to print information to.
        If not provided, the stdout channel will be used.
    """

    if name is not None:
        print("name: {:s}".format(name), file=file)
    print("shape: ", tensor.shape, file=file)
    print("dtype: ", tensor.dtype, file=file)
    print("device: ", tensor.device, file=file)
    print("mem: ", src.utils.memory.sprint_fancy_num_bytes(src.utils.memory.get_num_bytes(tensor)), file=file)
    if nl:
        print()



def normalize_image_tensor(img):
    """
    Transforms a C x W x H float tensor with values rescaled into the [0, 1] interval
        (per channel).

    :param img: torch.Tensor
        Original image tensor.
    
    :return: torch.Tensor
        Normalized image tensor.
    """


    img_flt = img.flatten(start_dim=1, end_dim=2)

    img_max = img_flt.max(dim=1).values[:, None, None]
    img_min = img_flt.min(dim=1).values[:, None, None]

    return (img - img_min) / (img_max - img_min)



def standardize_image_tensor(img):
    """
    Transforms a C x W x H float tensor with values standardized (per channel).

    :param img: torch.Tensor
        Original image tensor.
    
    :return: torch.Tensor
        Standardized image tensor.
    """


    img_flt = img.flatten(start_dim=1, end_dim=2)

    img_mean = img_flt.mean(dim=1)[:, None, None]
    img_std = img_flt.std(dim=1)[:, None, None]

    return (img - img_mean) / img_std
