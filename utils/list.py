def cutdown_list(my_list, ratio):
    """
    Takes a percentile subset of a list.

    :param my_list: list
        List to cut down.
    :param ratio: float
        Percentage of list to keep.

    :return: list
        Cut down list.
    """

    new_len = round(len(my_list) * ratio)
    return my_list[:new_len]
