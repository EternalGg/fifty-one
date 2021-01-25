import numpy as np
import activation_function
import optimizer
import modules
import mine_plot as mp


def cost_function(X,Y,theta):
    """
    :param X: train data (m,n)
    :param Y: train target (m,1)
    :param theta: parameter (n,1)
    :return: 1/m [(y * log(x*theta) + (1-y) * log(1 - x*theta)]
    """
    A = activation_function.sigmod(np.dot(X,theta))
    FIRST = Y * np.log(A)
    SECOND = (1 - Y) * np.log(1 - A)
    return np.sum(FIRST+SECOND) / len(X)


def fit(alpha, maxloop, epsilon, X, Y, module='', optimizers='logistic_bgd'):

    m, n = np.shape(X)
    theta = np.zeros((n, 1))
    print()
    count = 0
    converged = False

    thetas = {}
    error = np.inf
    errors = [cost_function(X,Y,theta)]
    for i in range(n):
        thetas[i] = [theta[i, 0], ]

    while count <= maxloop:
        if (converged):
            break
        count = count + 1


        for j in range(n):
            optimizer_dict = {'logistic_bgd': optimizer.logistic_bgd(X, j, theta, Y, m)}
            deriv = optimizer_dict[optimizers]
            thetas[j].append(theta[j, 0] - alpha * deriv)


        for j in range(n):
            theta[j, 0] = thetas[j][-1]

        error = cost_function(X,Y,theta)
        errors.append(error)
        if (abs(errors[-1])< epsilon):

            converged = True
    mp.show_linear_relation(X,theta)

    return theta, errors, thetas
