import numpy as np
import bridge
# import linear_regression


train_data =np.array([[1,1],
              [0,0]])

train_target = np.array([[1],
                [0]])

a,b,c = bridge.fit(train_data,train_target,'logistic_regression',0.01,0.01,5000,'','logistic_bgd')
print(train_target.shape)