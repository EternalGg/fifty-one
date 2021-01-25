import numpy as np
import activation_function as active

def h(theta, X):
    return np.dot(X, theta)


def mse(theta, X, Y):
    m = len(X)
    return np.sum(np.dot((h(theta,X)-Y).T, (h(theta,X)-Y)) / (2 * m))

def log_los(X,Y,theta):
    """
    :param X: train data (m,n)
    :param Y: train target (m,1)
    :param theta: parameter (n,1)
    :return: 1/m [(y * log(x*theta) + (1-y) * log(1 - x*theta)]
    """
    A = active.sigmod(np.dot(X,theta))
    FIRST = Y * np.log(A)
    SECOND = (1 - Y) * np.log(1 - A)
    return np.sum(FIRST+SECOND) / len(X)


def reg_log_loss(theta, X, Y, lamdba):
    """

    :param theta:
    :param X:
    :param Y:
    :return:
    """
    A = active.sigmod(np.dot(X, theta))
    FIRST = Y * np.log(A)
    SECOND = (1 - Y) * np.log(1 - A)
    reg = np.sum(np.power(theta[1:],2))*(lamdba/(2*len(X)))
    return np.sum(FIRST + SECOND) / len(X) + reg