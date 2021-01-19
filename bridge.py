import numpy as np
import matplotlib.pyplot as plt
import module as module
import numpy.matlib
import matplotlib.ticker as mtick
import linear_regression

model_dict = {'linear_regression': linear_regression}

def fit(X,Y,model='',alpha=0.01,maxloop=0.1,epsilon=0):
    a, b, c = model_dict[model].fit(alpha, maxloop , epsilon, X, Y)

    return a,b,c
# fig, ax = plt.subplots()
# for alpha in range(2):
#     ax.plot(np.arange(np.shape(c[0])[0]), c[alpha])
#     ax.legend()
#
# # ax.set(xlabel='iters',
# #        ylabel='cost',
# #        title='cost vs iters')
# plt.show()
# print(np.shape(train_data[:,0].reshape(3,1)))

# ax.scatter(np.dot(train_data[:,0].reshape(3,1),a[0]),np.dot(train_data[:,1].reshape(3,1),a[1]),c='r',marker='x',label='y=0')
# ax.scatter(train_data[:,0],train_data[:,1],c='b',marker='o',label='y=0')

# ax.plot(train_data[:,0],train_target)
# ax[1].plot

# ax.plot(np.dot(train_data[:,0].reshape(3,1),a[0])+np.dot(train_data[:,1].reshape(3,1),a[1]))
# ax.plot(
# fig , ax = plt.subplots()
# ax.plot(train_data[:,0],np.dot(train_data[:,0].reshape(3,1),a[0]))
# ax.plot(train_data[:,1],np.dot(train_data[:,1].reshape(3,1),a[1]))
# ax.legend()
# ax.legend()
# plt.show()
# def