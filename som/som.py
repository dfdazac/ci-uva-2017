import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
        self.energy = []
        self.energy_times = []

    def self_organize(self, data, iters=1000):
        """ Carries out self-organization in the SOM using data.
        Args:
            - data: An m by N matrix with data points in the columns.
        """        
        if iters <= 0:
            raise ValueError("iters must be positive")
        if data.shape[0] != self.m:
            raise ValueError("Incorrect data dimensions")

        # Initialize neuron weights from unique samples of the data
        indices = np.random.choice([i for i in range(data.shape[1])], self.n_units, replace=False)
        #self.neurons[...] = data[:, indices] # Initialize with samples
        #self.neurons = np.random.rand(self.m, self.h*self.w) # Randomly initialize all weights between 0 and 1
        min_data = np.min(data, axis=1)[:, np.newaxis]
        max_data = np.max(data, axis=1)[:, np.newaxis]
        self.neurons[...] = np.random.random((self.m, self.n_units))*(max_data - min_data) + min_data

        self.energy = []        
        
        # Algorithm constants
        r0 = self.map_radius 
        t1 = iters/np.log(self.map_radius)
        t2 = iters
        e0 = 0.5
        # Run!
        print("Self-organizing...")
        self._adapt(data, iters, r0, t1, t2, e0)       
        print("Done. Energy attained: {:.3f}".format(self.energy[-1]))

    def _adapt(self, data, n_iters, r0, t1, t2, e0):
        N = data.shape[1]
        prev_n = 0

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

            # The energy is sampled 10 times
            if (n - prev_n) >= n_iters/10:
                prev_n = n
                self.energy.append(self._calculate_energy(data))
        
        # Append last attained energy and calculate an epochs vector for it
        self.energy.append(self._calculate_energy(data))
        self.energy_times = np.linspace(0, 1, len(self.energy), endpoint=True) * n_iters
        
    def _calculate_energy(self, data):
        """ Calculates the energy from the data
        as the average minimum distance to the closest
        prototype.
        Args:
            - data: array, m by N
        Returns:
            - float, the average energy
        """
        average_energy = 0
        for i in range(data.shape[1]):
            x = data[:,i:i+1]
            distances = np.linalg.norm(self.neurons - x, axis=0)
            closest_i = np.argmin(distances)
            average_energy += distances[closest_i]
        return average_energy/data.shape[1]

    def plot_energy(self):
        """ Plots the history of the SOM energy during
        the self-organization process, if existent.                
        """
        if len(self.energy) > 0:
            plt.figure()
            plt.plot(self.energy_times, self.energy, "--bo")
            plt.title("Training Energy")
            plt.xlabel("Epochs")
            plt.grid()
            plt.show()
        else:
            print("No energy stored")

    def plot_hitmap(self, data):
        """ Plots the hit map of the SOM given the data.
        The area of the filled squares corresponds to the
        relative count of hits.
        Args:
            - data: array, m by N.
        """
        plt.figure()
        axis = plt.gca()

        # For each data points count the hits for the BMU
        rel_hits = np.zeros(self.n_units)
        for i in range(data.shape[1]):
            x = data[:,i:i+1]
            distances = np.linalg.norm(self.neurons - x, axis=0)
            closest_i = np.argmin(distances)
            rel_hits[closest_i] += 1
        rel_hits /= np.max(rel_hits)
        
        # Add grid and square for each neuron
        for i in range(self.h):
            for j in range(self.w):
                axis.add_patch(Rectangle((j, i), 1, 1, fill=False))
                side = rel_hits[np.ravel_multi_index((i, j), (self.h, self.w))]
                d = (1 - side)/2
                axis.add_patch(Rectangle((j+d, i+d), side, side, alpha=0.5))
        
        # Adjust visuals and plot
        plt.title("Hit Map")
        axis.axis("off")
        plt.xlim([-0.1,self.w+0.1])
        plt.ylim([-0.1,self.h+0.1])
        axis.set_aspect('equal', 'datalim')
        plt.show()


