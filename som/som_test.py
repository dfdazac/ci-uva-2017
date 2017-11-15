from som import SelfOrganizingMap
import numpy as np
import matplotlib.pyplot as plt

mu1 = [0,0]
mu2 = [0, 2]
mu3 = [2, 2]
mu4 = [2, 0]
sigma = [[0.01, 0], [0, 0.01]]
n = 100

data = np.concatenate((np.random.multivariate_normal(mu1, sigma, n).T, np.random.multivariate_normal(mu2, sigma, n).T, np.random.multivariate_normal(mu3, sigma, n).T, np.random.multivariate_normal(mu4, sigma, n).T), axis=1)
data = data[:, np.random.choice([i for i in range(data.shape[1])], data.shape[1],replace=False)]
print(data.shape)
plt.scatter(data[0,:], data[1,:])
plt.show()

som = SelfOrganizingMap(10, 10, 2)
print(som.neurons)
som.self_organize(data)
print(som.neurons)
plt.scatter(som.neurons[0,:], som.neurons[1,:])
plt.show()