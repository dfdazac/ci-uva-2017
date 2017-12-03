"""This example file shows how to use the SimpleESN class, in the scikit-learn
fashion. It is inspired by the minimalistic ESN example of Mantas Lukoševičius
"""
# Copyright (C) 2015 Sylvain Chevallier <sylvain.chevallier@uvsq.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
    data = loadtxt("data/MackeyGlass_fnc.txt")
    X = np.hstack((np.ones((len(data), 1)), data[:, 0:1]))
    Y = data[:, 1:2]
    train_length = 200
    test_length = 300
    
    X_train = X[:train_length]
    y_train = Y[:train_length]
    X_test = X[train_length:train_length+test_length]
    y_test = Y[train_length:train_length+test_length]

    # Simple training
    my_esn = SimpleESN(n_readout=50, n_features=X_train.shape[1],
        n_components=50, damping=1.0, weight_scaling=0.5)
    echo_train = my_esn.transform(X_train)
    regr = Ridge(alpha=0.001)
    regr.fit(echo_train, y_train)

    y_train_pred = regr.predict(echo_train)
    train_err = mean_squared_error(y_train, y_train_pred)
    
    echo_test = my_esn.transform(X_test)
    y_pred = regr.predict(echo_test)
    test_err = mean_squared_error(y_test, y_pred)
    
    fp = plt.figure(figsize=(12, 4))
    trainplot = fp.add_subplot(1, 2, 1)
    trainplot.plot(y_train, 'b', label="Target")
    trainplot.plot(y_train_pred, 'g', label="Prediction")
    plt.legend()
    trainplot.set_title('Training set (MSE {:.9f})'.format(train_err))
    testplot =  fp.add_subplot(1, 2, 2)
    testplot.plot(y_test, 'b', label='Target')
    testplot.plot(y_pred, 'g', label='Prediction')
    plt.legend()
    testplot.set_title('Test set (MSE {:.9f})'.format(test_err))
    plt.tight_layout(0.5)
    plt.show()
    

# Alpha MSE
# 1     1.47e-3
# 0.1   435e-6
# 0.01  39e-6
# 0.05  22e-6
# 0.01  11e-6
# 0     2e-6
