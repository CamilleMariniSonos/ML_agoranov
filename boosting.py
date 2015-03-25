"""
    @Stan
    This script enables to
    - load data
    - run gradient boosting and adaboost
    - compare their predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

# Enter file names
path = '../../Dreem/Dreem-So-Detection/data/kaggle/'
data_train = 'data_train.csv'
data_test = 'data_test.csv'
pred_train = 'labels_train.csv'
pred_test = 'labels_test.csv'

# loading
df = pd.read_csv(path + data_train, header=None)
x_train = df.values[1:, 1:].astype(np.float32)

df = pd.read_csv(path + data_test, header=None)
x_test = df.values[1:, 1:].astype(np.float32)

df_y = pd.read_csv(path + pred_train)
y_train = df_y['pred'].values

df_y = pd.read_csv(path + pred_test, header=None)
y_test = df_y.values[:, 1]


# Prediction algorithms
gbr = GradientBoostingRegressor(loss='ls',
                                n_estimators=10,
                                learning_rate=0.1,
                                verbose=10)

adb = AdaBoostRegressor(loss='linear',
                        n_estimators=50,
                        learning_rate=1.0,
                        verbose=10)

# Fitting and testing
gbr.fit(x_train, y_train)
s_gbr = gbr.score(x_test, y_test)

adb.fit(x_train, y_train)
s_adb = adb.score(x_test, y_test)

'''what about metrics ?'''

# Parameters for gbr
# loss function
# n_estimators
# learning rate
# subsample

# Parameters for adb
# loss
# n_estimators
# learning rate
