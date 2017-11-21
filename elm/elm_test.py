import numpy as np
import matplotlib.pyplot as plt
from elm import ExtremeLearningMachine

X = np.linspace(0, 2*np.pi, 50)[np.newaxis,:]
T = np.vstack((np.sin(X), np.cos(X)))

elm = ExtremeLearningMachine(1, 10, 2)
elm.train(X, T)

pred = elm.predict(X)
plt.plot(pred.T)
plt.plot(T.T)
plt.show()