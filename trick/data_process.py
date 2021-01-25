import numpy as np
import pandas as pd


def feature_mapping(x1,x2,power):
    data = {}

    for i in np.arange(power+1):
        for j in np.arange(i+1):
            data['F{}{}'.format(i-j,j)] = np.power(x1,i-j) * np.power(x2,j)
    return pd.DataFrame(data)

