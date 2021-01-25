import numpy as np
import modules
import activation_function


def bgd(X, tern, theta, Y, m):
    deriv = np.dot(X[:, tern].T, (modules.h(theta, X) - Y)).sum() / m
    return deriv

def logistic_bgd(X, tern, theta, Y, m):
    print(np.shape(X))

    print(np.shape(theta))
    deriv = np.dot(X[:, tern].T, (activation_function.sigmod(np.dot(X, theta)) - Y)).sum() / m
    print(deriv)
    return deriv