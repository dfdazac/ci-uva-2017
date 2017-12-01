from numpy import loadtxt
from som import SelfOrganizingMap
import pickle

print("Loading data...")
data = loadtxt("../data/edges/total.csv").T

som = SelfOrganizingMap(5, 5, data.shape[0])
som.self_organize(data, iters=40000)
som.plot_energy()
som.plot_hitmap(data)
som.plot_umatrix()
som.plot_dendrogram()

pickle.dump(som, open("som.p", "wb"))