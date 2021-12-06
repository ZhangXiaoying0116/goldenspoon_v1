import os
import pickle

def pickle_cache(filename, callback):
    if os.path.isfile(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)
    result = callback()
    with open(filename, 'wb') as fh:
        pickle.dump(result, fh)
    return result

