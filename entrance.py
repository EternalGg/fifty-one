import numpy as np
import bridge
# import linear_regression


train_data =np.array([[2,4],
              [2,2],
              [3,3]])
train_target = np.array([[4],
                [3],
                [4.5]])

a,b,c = bridge.fit(train_data,train_target,'linear_regression',0.01,0.01,5000)
