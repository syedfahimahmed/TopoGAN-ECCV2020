import os
import numpy as np

def check_dir(path):
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)

def rescale(array, min_v = 0, max_v = 1):
    if np.max(array) - np.min(array) == 0:
        return np.zeros(array.shape,dtype=np.float32)
    rescaled= (array - np.min(array)) / (np.max(array) - np.min(array))
    return rescaled * (max_v - min_v) + min_v