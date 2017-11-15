import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap:
    def __init__(self, h, w, m):
        """ A self organizing map with dimensions h by w
        and input dimensionality n. Note that a map of
        this dimensions creates h * w neurons, each with
        m weights.
        """
        self.h = h
        self.w = w
        self.m = m
        self.n_units = h * w
        self.map_radius = max(h-1, w-1)/2
        self.neurons = np.empty((m, h * w))
        self.map_pos = np.array([(i,j) for i in range(h) for j in range(w)])

    def self_organize(self, data, monitor=False):
        """ Carries out self-organization in the SOM using data.
        Args:
            - data: An m by N matrix with data points in the columns.
        """
        N = data.shape[1]
        # Check if data is correct
        if N < self.n_units:
            raise ValueError("Insufficient data for self-organization")
        if data.shape[0] != self.m:
            raise ValueError("Incorrect data dimensions")

        # Initialize neuron weights from unique samples of the data
        indices = np.random.choice([i for i in range(N)], self.n_units, replace=False)
        self.neurons[...] = data[:, indices]

        # Algorithm constants
        organizing_iters = 1000
        convergence_iters = 500 * self.n_units
        t_1 = 1000/np.log(self.map_radius)
        t_2 = 1000
        r_0 = 0.1
        
        for i in range(N):
            print(i)
            x = data[:,i:i+1]
            # 1) Competitive process: find coordinate of the winning neuron
            winner_i = np.unravel_index(np.argmax(x.T @ self.neurons), (self.h, self.w))
            plt.figure()
            plt.scatter(self.neurons[0,:], self.neurons[1,:])
            plt.scatter(data[0,i], data[1,i])
            plt.title("Before")
            plt.savefig(str(i) + "a")
            # Self-organizing phase
            for n in range(organizing_iters):
                # 2) Cooperative process: calculate neighborhood radius
                # and select neurons within it                
                radius = self.map_radius * np.exp(-n/t_1)
                distances = np.linalg.norm(self.map_pos - winner_i, axis=1)
                selected = distances <= radius
                neighborhood = np.exp(-distances[selected]**2 / (2*radius**2))
                
                # 3) Adaptive process: update weights of selected neurons
                r = r_0 * np.exp(-n/t_2)
                self.neurons[:, selected] -= r * neighborhood * (self.neurons[:, selected] - x)

            # Convergence phase
            for n in range(convergence_iters):
                self.neurons[:, selected] -= 0.01 * neighborhood * (self.neurons[:, selected] - x)
            plt.figure()
            plt.scatter(self.neurons[0,:], self.neurons[1,:])
            plt.scatter(data[0,i], data[1,i])
            plt.title("After")
            plt.savefig(str(i) + "b")

            if i == 10:
                return

if __name__ == '__main__':
    som = SelfOrganizingMap(20, 20, 3)
    data = np.random.randint(0, 9, (3,1000))
    som.self_organize(data)