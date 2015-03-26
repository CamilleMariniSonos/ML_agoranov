"""
    @Stan
    This script enables to
    - load data
    - run gradient boosting and adaboost
    - compare their predictions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from matplotlib.widgets import Slider, Button


# Loading data
json_data = open('data.json').read()
data = json.loads(json_data)
X = np.asarray(data['X'])
Y = np.asarray(data['Y'])


# Splitting into train and test set
n = int(2. / 3. * len(Y))
x_train = X[0:n, :]
y_train = Y[0:n]

x_test = X[n::, :]
y_test = Y[n::]

m = len(y_test)

print x_train.shape
print y_train.shape

print x_test.shape
print y_test.shape

# Prediction algorithms
gbr = GradientBoostingRegressor(loss='ls',
                                n_estimators=20,
                                learning_rate=0.1,
                                max_depth=10,
                                subsample=0.8,
                                verbose=10)

# # Performing a grid search to get the best parameters
# # May take some time
# grid = dict(loss=['ls'],
#             n_estimators=np.arange(5, 50, 5),
#             learning_rate=np.linspace(0.01, 0.1, num=10),
#             max_depth=range(1, 10),
#             subsample=np.linspace(0.01, 1.0, num=10),
#             verbose=[False])

# gd = GridSearchCV(gbr, param_grid=grid, cv=5, scoring='mean_squared_error')
# gd.fit(x_train, y_train)

# print("Best parameters set found on development set:")
# print()
# print(gd.best_estimator_)

# best = gd.best_estimator_
# best.fit(x_train, y_train)
# y_best = best.predict(x_test)

# interactive plot


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)
t = np.arange(0.0, 1.0, 0.001)
time = range(0, m)
plt.plot(time, y_test, 'o-', color="r",
         label="True", linewidth=2.0)
# plt.plot(time, y_best, 'o-', color="g",
#          label="Best", linewidth=2.0)

# initial parameters
n0 = 10
lr0 = 0.1
md0 = 3
sb0 = 1.0
p0 = dict(loss='ls',
          n_estimators=n0,
          learning_rate=lr0,
          max_depth=md0,
          subsample=sb0,
          verbose=False)
gbr.set_params(**p0)
gbr.fit(x_train, y_train)
y_pred = gbr.predict(x_test)

# initial plot
l, = plt.plot(time, y_pred, 'o-', color="b",
              label="Pred", linewidth=2.0)

# sliders positions
axcolor = 'lightgoldenrodyellow'
ax_nest = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
ax_lr = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
ax_md = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg=axcolor)
ax_sb = plt.axes([0.25, 0.25, 0.65, 0.03], axisbg=axcolor)

# sliders definition
snest = Slider(ax_nest, 'n estimators', 1, 50, valinit=n0, valfmt=u'%1d')
slr = Slider(ax_lr, 'learning rate', 0.01, 0.1, valinit=lr0, valfmt=u'%1.2f')
smd = Slider(ax_md, 'max depth', 1, 10, valinit=md0, valfmt=u'%1d')
ssb = Slider(ax_sb, 'Subsample', 0.1, 1.0, valinit=sb0, valfmt=u'%1.2f')


def update(val):
    '''
    This function enables to update the plot
    '''
    nest = int(snest.val)
    lr = slr.val
    md = int(smd.val)
    sb = ssb.val
    p = dict(loss='ls',
             n_estimators=nest,
             learning_rate=lr,
             max_depth=md,
             subsample=sb,
             verbose=False)
    gbr.set_params(**p)
    gbr.fit(x_train, y_train)
    y_pred = gbr.predict(x_test)
    l.set_ydata(y_pred)
    fig.canvas.draw_idle()
snest.on_changed(update)
slr.on_changed(update)
smd.on_changed(update)
ssb.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    '''
    This function enables to reset sliders values
    '''
    snest.reset()
    slr.reset()
    smd.reset()
    ssb.reset()
button.on_clicked(reset)

plt.show()
