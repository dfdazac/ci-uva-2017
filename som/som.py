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
        self.map = np.empty((h, w, m))
        self.map_idx = np.array([[(i,j) for j in range(w)] for i in range(h)])

    def self_organize(self, data):
        """ Carries out self-organization in the SOM using data.
        Args:
            - data: An N by m matrix with data points in the columns.
        """
        N = len(data)
        # Check if data is correct
        if N < self.n_units:
            raise ValueError("Insufficient data for self-organization")
        if data.shape[1] != self.m:
            raise ValueError("Incorrect data dimensions")

        # Initialize weights from unique samples of the data
        indices = np.random.choice([i for i in range(N)], self.n_units, replace=False)
        self.map[...] = data[indices].reshape((self.h, self.w, self.m))

        # Algorithm constants
        organizing_iters = 1000
        convergence_iters = 500 * self.n_units
        t_1 = 1000/np.log(self.map_radius)
        t_2 = 1000
        r_0 = 0.1
        
        for x in data:
            # 1) Competitive process: find index of the winning neuron
            winner_i = np.unravel_index(np.argmax(self.map @ x[:, np.newaxis]), (self.h, self.w))

            # Self-organizing phase
            for n in range(organizing_iters):
                # 2) Cooperative process: calculate neighborhood radius
                # and select neurons within it
                print("Winner:", winner_i)
                radius = self.map_radius * np.exp(-n/t_1)
                distances = np.linalg.norm(self.map_idx - winner_i, axis=2)
                selected = distances <= radius
                neighborhood = np.exp(-distances[selected]**2 / (2*radius**2))

                # If there are S selected neurons, neighborhood contains S
                # elements. The next step (equation 9.13) should compute
                # (x - w) with w being the S selected neurons (accessed using
                # the selected array). The result is multiplied element-wise
                # by neighborhood.
                
                # 3) Adaptive process: update weights
                self.map[selected] = 0
                plt.imshow(np.sum(self.map, axis=2))
                plt.show()
                return

            return



            # Convergence phase
            for t in range(convergence_iters):
                pass

if __name__ == '__main__':
    som = SelfOrganizingMap(20, 20, 3)
    data = np.random.randint(0, 9, (1000,3))
    som.self_organize(data)