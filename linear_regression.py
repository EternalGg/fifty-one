import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import matplotlib.ticker as mtick
import module

def J(theta, X, Y):
    m = len(X)
    return np.sum(np.dot((module.h(theta,X)-Y).T, (module.h(theta,X)-Y)) / (2 * m))


def fit(alpha, maxloop, epsilon, X, Y):
    m, n = np.shape(X)

    theta = np.zeros((n, 1))  # 参数theta全部初始化为0

    count = 0  # 记录迭代轮次
    converged = False  # 是否已经收敛的标志
    error = np.inf  # 当前的代价函数值
    errors = [J(theta, X, Y), ]  # 记录每一次迭代得代价函数值

    thetas = {}
    for i in range(n):
        thetas[i] = [theta[i, 0], ]  # 记录每一个theta j的历史更新

    while count <= maxloop:
        if (converged):
            break
        count = count + 1


        for j in range(n):

            deriv = np.dot(X[:, j].T, (module.h(theta, X) - Y)).sum() / m  #梯度等于 x
            thetas[j].append(theta[j, 0] - alpha * deriv)


        for j in range(n):
            theta[j, 0] = thetas[j][-1]

        error = J(theta, X, Y)
        errors.append(error)

        if (abs(errors[-1])< epsilon):
            converged = True
    show_linear_relation(X,theta)

    return theta, errors, thetas

def show_linear_relation(X,THETA):
    fig, ax = plt.subplots()
    shape = np.shape(X[:, 0])[0]
    print(shape)
    for i in range(shape):
        ax.plot(X[:, 0], np.dot(X[:, 0].reshape(shape, 1), THETA[0]))
        ax.plot(X[:, 1], np.dot(X[:, 1].reshape(shape, 1), THETA[1]))
        ax.legend()

    plt.show()