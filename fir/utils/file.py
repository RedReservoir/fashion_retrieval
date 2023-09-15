import os
import uuid



def generate_unused_filename(dir="."):
    """
    Finds a random unused filename.

    :param dir: str, default="."
        Target directory in which to check for files.
        Current directory by default.
    
    :return: str
        A filename not currently used.
    """

    filename = os.path.join(dir, str(uuid.uuid4()))
    while os.path.exists(filename):
        filename = os.path.join(dir, str(uuid.uuid4()))
    
    return filename