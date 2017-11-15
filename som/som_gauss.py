from som import SelfOrganizingMap
import numpy as np
import matplotlib.pyplot as plt

mu1 = [-10,-10]
mu2 = [-10, -10]
mu3 = [10, 10]
mu4 = [10, 10]
sigma = [[0.1, 0], [0, 0.1]]
n = 400

data = np.concatenate((np.random.multivariate_normal(mu1, sigma, n).T, np.random.multivariate_normal(mu2, sigma, n).T, np.random.multivariate_normal(mu3, sigma, n).T, np.random.multivariate_normal(mu4, sigma, n).T), axis=1)
data = data[:, np.random.choice([i for i in range(data.shape[1])], data.shape[1],replace=False)]
#print(data.shape)
#plt.scatter(data[0,:], data[1,:])
#plt.show()

som = SelfOrganizingMap(15, 15, 2)

som.self_organize(data, iters=2000)
som.plot_energy()

