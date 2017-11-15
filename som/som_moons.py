from sklearn import datasets
import matplotlib.pyplot as plt
from som import SelfOrganizingMap

data, _ = datasets.make_moons(1000, noise=0.1, shuffle=True)
plt.figure()
plt.scatter(data[:,0], data[:,1])

som = SelfOrganizingMap(10, 10, 2)
som.self_organize(data.T, iters=1000)
plt.figure()
plt.scatter(som.neurons[0,:], som.neurons[1,:])

som.plot_energy()
som.plot_hitmap(1)