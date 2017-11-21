from sklearn import datasets
import matplotlib.pyplot as plt
from som import SelfOrganizingMap

data, _ = datasets.make_moons(1000, noise=0.05, shuffle=True)

som = SelfOrganizingMap(20, 20, 2)

som.self_organize(data.T, iters=8000)

#som.plot_energy()
#som.plot_hitmap(data.T)
#som.plot_2dmap(data.T)
#som.plot_umatrix()
som.plot_dendrogram()