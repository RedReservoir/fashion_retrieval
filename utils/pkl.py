import pickle as pkl


def pickle_test(obj):
    """
    Checks if an object is pickable.

    :param obj: any
        Object to pickle.
    
    :return: str or bool
        Returns True if object is pickable.
        Returns a string with the exception type and message otherwise.
    """
    
    try:
        pkl.dumps(obj)
    except Exception as ex:
        return "{:s}: {:s}".format(type(ex).__name__, str(ex))
    return True
