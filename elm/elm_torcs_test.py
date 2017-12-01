import pickle
from elm import ExtremeLearningMachine
import matplotlib.pyplot as plt
import numpy as np

def test(inputs, targets):
    elm = pickle.load(open("elm_driver.p", "rb"))    
    
    Y = elm.predict(inputs.T)

    plt.plot(targets, label="Target")
    plt.plot(Y.T, label="Prediction")
    plt.legend()
    plt.show()

def steer_smooth():
    data = np.loadtxt("../data/new_data.txt")

    inputs = data[:, 0:1]
    targets = data[:, 1:2]
    test(inputs, targets)

steer_smooth()