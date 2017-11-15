from sklearn import datasets
from som import SelfOrganizingMap
import matplotlib.pyplot as plt

data, _ = datasets.make_moons(1000, noise=0.1, shuffle=True)
som = SelfOrganizingMap(10, 10, 2)
som.self_organize(data.T)
plt.scatter(som.neurons[0,:], som.neurons[1,:])
plt.show()