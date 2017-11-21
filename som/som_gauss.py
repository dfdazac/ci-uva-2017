from som import SelfOrganizingMap
import numpy as np
import matplotlib.pyplot as plt

mu1 = [-5,-5]
mu2 = [5, -5]
mu3 = [5, 5]
mu4 = [-5, 5]
sigma = [[1, 0], [0, 1]]
n = 200

data = np.concatenate((np.random.multivariate_normal(mu1, sigma, n).T, np.random.multivariate_normal(mu2, sigma, n).T, np.random.multivariate_normal(mu3, sigma, n).T, np.random.multivariate_normal(mu4, sigma, n).T), axis=1)
data = data[:, np.random.choice([i for i in range(data.shape[1])], data.shape[1],replace=False)]

som = SelfOrganizingMap(15, 15, 2)
som.self_organize(data, iters=5000)
#som.plot_energy()
#som.plot_hitmap(data)
#som.plot_2dmap(data)
#som.plot_umatrix()
som.plot_dendrogram()