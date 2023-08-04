def chain_get(my_dict, *args, default=None):

    key_list = list(args)
    curr_dict = my_dict

    for key in key_list:

        try:
            curr_dict = curr_dict[key]
        except:
            return default

    return curr_dict
