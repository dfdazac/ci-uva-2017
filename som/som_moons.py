from sklearn import datasets
import matplotlib.pyplot as plt
from som import SelfOrganizingMap

data, _ = datasets.make_moons(1000, noise=0.1, shuffle=True)

som = SelfOrganizingMap(10, 10, 2)
som.self_organize(data.T, iters=4000)

som.plot_energy()
som.plot_hitmap(data.T)
som.plot_2dmap(data.T)