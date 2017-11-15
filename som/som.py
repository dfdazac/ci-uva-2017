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

    def self_organize(self, data):
        """ Carries out self-organization in the SOM using data.
        Args:
            - data: An m by N matrix with data points in the columns.
        """        
        if data.shape[0] != self.m:
            raise ValueError("Incorrect data dimensions")

        # Initialize neuron weights from unique samples of the data
        indices = np.random.choice([i for i in range(data.shape[1])], self.n_units, replace=False)
        self.neurons[...] = data[:, indices]

        
        # Algorithm constants
        print("Running self-organizing phase...")
        organizing_iters = 1000
        r0 = self.map_radius      
        t1 = organizing_iters/np.log(self.map_radius)
        t2 = organizing_iters
        e0 = 0.1
        self._adapt(data, organizing_iters, r0, t1, t2, e0, "a")
        
        print("Running convergence phase...")
        convergence_iters = 500 * self.n_units
        r0 = 1
        t1 = convergence_iters/0.7 # -0.7 = ln(0.5), 0.5 is the final value
        t2 = convergence_iters/2.3 # -2.3 = ln(0.1), 0.1 is the final value
        e0 = 0.01
        self._adapt(data, convergence_iters, r0, t1, t2, e0, "b")        

    def _adapt(self, data, n_iters, r0, t1, t2, e0, title):
        N = data.shape[1]
        
        for n in range(n_iters):
            # Sample data point
            i = np.random.randint(0, N)
            x = data[:,i:i+1]

            # 1) Competitive process: find coordinate of the winning neuron
            winner_i = np.unravel_index(np.argmin(np.linalg.norm(self.neurons - x, axis=0)), (self.h, self.w))

            # 2) Cooperative process: calculate neighborhood radius
            # and select neurons within it                
            radius = r0 * np.exp(-n/t1)
            distances = np.linalg.norm(self.map_pos - winner_i, axis=1)
            selected = distances <= radius
            neighborhood = np.exp(-distances[selected]**2 / (2*radius**2))
            
            # 3) Adaptive process: update weights of selected neurons
            e = e0 * np.exp(-n/t2)
            self.neurons[:, selected] -= e * neighborhood * (self.neurons[:, selected] - x)
        
