def chain_get(my_dict, *args, default=None):
    """
    Accesses a dict object repeatedly using dict.get() multiple times.
    Accessing stops whenever a key is not found.

    :param my_dict: dict
        Dict to access.
    :param *args:
        Keys with which to access the dict.
    :param default: any, default=None
        Default return object.
    
    :return: any
        The accessed dict value or the default return object if not found.
    """
    
    key_list = list(args)
    curr_dict = my_dict

    for key in key_list:

        try:
            curr_dict = curr_dict[key]
        except:
            return default

    return curr_dict
