from simple_esn import SimpleESN
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from numpy import loadtxt, atleast_2d
import matplotlib.pyplot as plt
from pprint import pprint
from time import time
import numpy as np

np.random.seed(42)

if __name__ == '__main__':
    data = loadtxt("../data/center_steering/center_steering.txt")
    X = data[:, 0:1]
    Y = data[:, 1:2]
    split_idx = int(0.7 * len(X))
    
    X_train = X[:split_idx]
    y_train = Y[:split_idx]
    X_test = X[split_idx:]
    y_test = Y[split_idx:]

    # Simple training
    my_esn = SimpleESN(n_readout=100, n_components=100,
                       damping=0.5, weight_scaling=1.0)
    echo_train = my_esn.fit_transform(X_train)
    regr = Ridge(alpha=0.0)
    regr.fit(echo_train, y_train)
    foo = regr.predict(my_esn.transform(X_train))
    train_err = mean_squared_error(y_train, foo)
    
    echo_test = my_esn.transform(X_test)
    y_true, y_pred = y_test[300:], regr.predict(echo_test)[300:]
    err = mean_squared_error(y_true, y_pred)
    
    fp = plt.figure(figsize=(12, 4))
    trainplot = fp.add_subplot(1, 2, 1)
    trainplot.plot(y_train, 'b')
    trainplot.plot(foo, 'g')
    trainplot.set_title('Training (MSE %0.6f)' % train_err)
    testplot =  fp.add_subplot(1, 2, 2)
    testplot.plot(y_true, 'b', label='test signal')
    testplot.plot(y_pred, 'g', label='prediction')
    testplot.set_title('Validation (MSE %0.6f)' % err)
    plt.tight_layout(0.5)
    plt.show()
    
# Smoothed data, 200 components
# Alpha MSE
# 1     1.407e-3
# 0.1   1.152e-3
# 0.01  1.013e-3
# 0.005 959e-6
# 0.001 856e-6

# Unsmoothed data, 1000 components
# Alpha MSE
# 0.01  601e-6
# 0.005 582e-6
# 0.001 561e-6

# Same but removed bias
# 0.001 525e-6
