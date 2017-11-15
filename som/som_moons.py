from sklearn import datasets
from som import SelfOrganizingMap

data, _ = datasets.make_moons(1000, noise=0.1, shuffle=True)

som = SelfOrganizingMap(10, 10, 2)
som.self_organize(data.T, iters=500)

som.plot_energy()
