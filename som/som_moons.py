from sklearn import datasets
import matplotlib.pyplot as plt
from som import SelfOrganizingMap
from time import sleep

data, _ = datasets.make_moons(1000, noise=0.05, shuffle=True)
data = data.T

som = SelfOrganizingMap(20, 20, 2)

som.self_organize(data, iters=4000)

#som.plot_energy()
#som.plot_hitmap(data)
#som.plot_2dmap(data)
#som.plot_umatrix()
#som.plot_dendrogram()
#plt.show()

plt.ion()
fig = plt.figure()
prev_batch = None
for i in range(data.shape[1]):
    x = data[:, i:i+1]
    prev_batch = som.plot_closer(x, figure=fig, patch=prev_batch)
    plt.show() 
    plt.pause(0.0001)
