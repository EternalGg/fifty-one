import numpy as np
import trick.activation_function as active
import trick.optimizer as optimizer
import trick.modules as modules
import trick.mine_plot as mp




def fit(alpha, maxloop, epsilon, X, Y, module='', optimizers='logistic_bgd'):

    m, n = np.shape(X)
    theta = np.zeros((n, 1))
    print()
    count = 0
    converged = False
    model_dict = {'log_los': modules.log_los(theta, X, Y),
                  'reg_log_loss' : modules.reg_log_loss(theta,X,Y)
                  }
    thetas = {}
    error = np.inf
    errors = [model_dict[module]]
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

        error = model_dict[module]
        errors.append(error)
        if (abs(errors[-1])< epsilon):

            converged = True
    mp.show_linear_relation(X,theta)

    return theta, errors, thetas
