from numpy import loadtxt
from som import SelfOrganizingMap

all_data = loadtxt("alpine-1.csv", delimiter=",", skiprows=1)

# The sensor data starts at column 5
data = all_data[:, 5:].T

som = SelfOrganizingMap(30, 40, data.shape[0])
som.self_organize(data, iters=6000000)
som.plot_energy()
som.plot_umatrix()
som.plot_dendrogram()