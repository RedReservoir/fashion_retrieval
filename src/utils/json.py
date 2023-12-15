import json



def save_json_dict(
        json_dict,
        json_filename,
        indent=4
        ):
    """
    Saves a dict object to a JSON file.

    :param json_dict: dict
        Dict object to save.
    :param json_filename: str
        Name of the JSON file.
    :param indent: int, default=4
        Number of spaces for indentation.
    """

    with open(json_filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=indent)



def load_json_dict(
        json_filename
        ):
    """
    Loads a dict object from a JSON file.

    :param json_filename: str
        Name of the JSON file.

    :return: dict
        Loaded dict object.
    """

    with open(json_filename, 'r') as json_file:
        json_dict = json.load(json_file)

    return json_dict
