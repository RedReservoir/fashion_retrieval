import sys
import torch

import utils.mem


def print_tensor_info(tensor, name, out_stream=sys.stdout):

    print("name: {:s}".format(name), file=out_stream)
    print("  shape: ", tensor.shape, file=out_stream)
    print("  dtype: ", tensor.dtype, file=out_stream)
    print("  device: ", tensor.device, file=out_stream)
    print("  mem: ", utils.mem.sprint_fancy_num_bytes(utils.mem.get_num_bytes(tensor)), file=out_stream)
