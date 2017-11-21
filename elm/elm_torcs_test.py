import pickle
from elm import ExtremeLearningMachine
import matplotlib.pyplot as plt
import numpy as np

elm = pickle.load(open("elm_driver.p", "rb"))

data = np.loadtxt("../data/aalborg.csv", skiprows=1, delimiter=",").T

T = data[:3, :]
X = data[3:, :]
Y = elm.predict(X)

plt.subplot(2, 1, 1)
plt.plot(T[2,:].T)
plt.subplot(2, 1, 2)
plt.plot(Y[2,:].T)
plt.show()