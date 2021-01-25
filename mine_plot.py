import matplotlib.pyplot as plt
import numpy as np
from turtle import *


def show_linear_relation(X,THETA):
    fig, ax = plt.subplots()
    shape = np.shape(X[:, 0])[0]
    print(shape)
    for i in range(shape):
        ax.plot(X[:, 0], np.dot(X[:, 0].reshape(shape, 1), THETA[0]))
        ax.plot(X[:, 1], np.dot(X[:, 1].reshape(shape, 1), THETA[1]))
        ax.legend()

    plt.show()

def zmy():
    def curvemove():
        for i in range(100):
            right(2)
            forward(2)
    color('red', 'pink')
    begin_fill()
    left(140)
    forward(111.65)
    curvemove()
    left(120)
    curvemove()
    forward(111.65)
    goto(-30, 70)
    write("曾梦圆", font=("arial", 20, "normal"))
    end_fill()
    done()