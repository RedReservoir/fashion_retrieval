def compute_dgi_ratio(
        current_epoch_num,
        dgi_num_epochs,
        dgi_init_ratio=None
        ):
    """
    Computes dataset gradual increase (dgi) ratio.
    
    Example with dgi_num_epochs = 5 and dgi_init_ratio = 0.25
        epoch  1 | dgi_ratio: 0.2500
        epoch  2 | dgi_ratio: 0.3000
        epoch  3 | dgi_ratio: 0.4000
        epoch  4 | dgi_ratio: 0.6000
        epoch  5 | dgi_ratio: 1.0000
        ...

    :param current_epoch_num: int
        Current epoch number.
    :param dgi_num_epochs: int
        Number of epochs until full dataset size is reached.
    :param dgi_init_ratio: float, optional
        Initial dgi ratio (at epoch 1).
        If not provided, it will be autoselected such that dataset size is double on each epoch.
    
    :return: float
        Current dgi ratio.
    """

    if dgi_init_ratio is None:
        dgi_init_ratio = 2 ** (-dgi_num_epochs + 1)

    if current_epoch_num >= dgi_num_epochs:
        dgi_ratio = 1
    else:
        num_dgi_segments = (2 ** (dgi_num_epochs - 1)) - 1
        curr_num_dgi_segments = (2 ** (current_epoch_num - 1)) - 1
        dgi_ratio = dgi_init_ratio + ((1 - dgi_init_ratio) * (curr_num_dgi_segments / num_dgi_segments))

    return dgi_ratio