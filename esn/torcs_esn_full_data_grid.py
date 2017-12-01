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
    data = loadtxt("../data/all_sensors_all_controls.csv",
        delimiter=",", skiprows=1)

    X = data[:, 3:]
    Y = data[:, 2]
    split_idx = int(0.7 * len(X))
    
    X_train = X[:split_idx]
    y_train = Y[:split_idx]
    X_test = X[split_idx:]
    y_test = Y[split_idx:]

    count = 0
    components = [100] #[10, 20, 40, 80, 100, 200, 400]
    dampings = [0.1, 0.2, 0.5, 0.75, 1]
    scalings = [0.6, 0.7, 0.9, 1.0, 1.2]
    trials = 10
    n_trials = trials * len(components) * len(dampings) * len(scalings)

    str_format = "{:d}/{:d} Components: {:d} - Damping: {:.2f} - Scaling: {:.2f} - Validation MSE: {:.6f}"

    for n_components in components:
        for damping in dampings:
            for weight_scaling in scalings:
                for i in range(trials):
                    count += 1

                    # Simple training
                    my_esn = SimpleESN(n_readout=n_components, n_components=n_components,
                                       damping=damping, weight_scaling=weight_scaling)
                    echo_train = my_esn.fit_transform(X_train)
                    regr = Ridge()
                    regr.fit(echo_train, y_train)
                    #foo = regr.predict(my_esn.transform(X_train))
                    
                    echo_test = my_esn.transform(X_test)
                    y_true, y_pred = y_test, regr.predict(echo_test)
                    err = mean_squared_error(y_true, y_pred)
                    
                    print(str_format.format(count, n_trials, n_components, damping, weight_scaling, err))
    
    # fp = plt.figure(figsize=(12, 4))
    # trainplot = fp.add_subplot(1, 2, 1)
    # trainplot.plot(y_train, 'b')
    # trainplot.plot(foo, 'g')
    # trainplot.set_title('Some training signal')
    # testplot =  fp.add_subplot(1, 2, 2)
    # testplot.plot(y_test, 'b', label='test signal')
    # testplot.plot(y_pred, 'g', label='prediction')
    # testplot.set_title('Prediction (MSE %0.6f)' % err)
    # plt.tight_layout(0.5)
    # plt.show()
    

