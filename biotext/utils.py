"""
Various generic utilities
"""

import pickle
import pathlib

def resolve_path(path, filename):
    "resolve path with pathlib"

    try:
        path=pathlib.Path(path)
        path_full=path.joinpath(filename)
    except:
        path_full=path + filename

    return path_full

def pickle_save(obj=None, path=None):
    "Function to objects to pickle"

    with open(path, 'wb') as output:
        pickle.dump(obj, output)

def pickle_load(path=None):
    "Function for loading pickle files"

    with open(path, 'rb') as f:
        return pickle.load(f)

