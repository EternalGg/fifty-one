import numpy as np

def sigmod(z):
    """
    :param z: X * theta
    :return: 1 / 1+e^-z
    """
    return 1 / (1 + np.exp(-z))