import numpy as np


def h(theta, X):
    return np.dot(X, theta)


def mse(theta, X, Y):
    m = len(X)
    return np.sum(np.dot((h(theta,X)-Y).T, (h(theta,X)-Y)) / (2 * m))