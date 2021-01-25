import numpy as np
import modules
import activation_function


def bgd(X, tern, theta, Y, m):
    deriv = np.dot(X[:, tern].T, (modules.h(theta, X) - Y)).sum() / m
    return deriv

def logistic_bgd(X, tern, theta, Y, m):
    deriv = np.dot(X[:, tern].T, (activation_function.sigmod(np.dot(X, theta)) - Y)).sum() / m
    print(deriv)
    return deriv

def reg_logistic_bgd(X, tern, theta, Y, m ,lamdba):
    reg = theta[1:]*(lamdba/len(X))
    reg = np.insert(reg,0,values=0,axis=0)

    deriv = np.dot(X[:, tern].T, (activation_function.sigmod(np.dot(X, theta)) - Y)).sum() / m - reg
    print(deriv)
    return deriv