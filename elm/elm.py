import numpy as np
from scipy.special import expit

class ExtremeLearningMachine:
    def __init__(self, n_input, n_hidden, n_output):
        """ Initializes an ELM given the parameters.
        Args:
            - n_input (int): number of input units
            - n_hidden (int): number of hidden units
            - n_output (int): number of output units
        """
        self.n = n_input
        self.h = n_hidden
        self.m = n_output

        # Initialization heuristic by Andrew Ng
        e_1 = np.sqrt(6)/np.sqrt(n_input + n_hidden)
        e_2 = np.sqrt(6)/np.sqrt(n_hidden + n_output)
        self.W_1 = np.random.uniform(-e_1, e_1, (n_input, n_hidden))
        self.b_1 = np.random.uniform(-e_1, e_1, (n_hidden, 1))
        self.W_2 = np.empty((n_hidden, n_output))

        self.data_mean = 0
        self.data_std = 0

    def _check_data(self, X, T):
        """ Checks consistency of the dimensions in feature and target data
        n and m are the number of input and output units, respectively,
        and N is the number of samples.
        Args:
            - X (n by N array): sample features
            - T (m by N array): sample targets
        """
        if X.shape[0] != self.n:
            raise ValueError("Input data dimension is incorrect. Got {:d}, expected {:d}".format(X.shape[0], self.n))
        if T.shape[0] != self.m:
            raise ValueError("Target data dimension is incorrect. Got {:d}, expected {:d}".format(T.shape[0], self.m))

    def mse(self, X, T):
        """ Calculates the Mean Square Error of the ELM for the given data.
        n and m are the number of input and output units, respectively,
        and N is the number of samples.
        Args:
            - X (n by N array): sample features
            - T (m by N array): sample targets
        Returns:
            - float, the MSE
        """
        self._check_data(X, T)

        n_samples = X.shape[1]
        T_pred = self.predict(X)
        error = np.sum(np.linalg.norm(T_pred - T, axis=0))
        return error/n_samples

    def train(self, X, T):
        """ Trains the ELM with the given data.
        n and m are the number of input and output units, respectively,
        and N is the number of samples.
        Args:
            - X (n by N array): sample features
            - T (m by N array): sample targets
        """
        self._check_data(X, T)
        # Normalize data
        self.data_mean = X.mean(axis=1)[:, np.newaxis]
        self.data_std = X.std(axis=1)[:, np.newaxis]
        X_n = (X - self.data_mean)/self.data_std

        # Calculate hidden layer output
        H = expit(self.W_1.T @ X_n + self.b_1)
        # Find the least squares solution for the output weights
        self.W_2[...] = np.linalg.inv(H @ H.T) @ H @ T.T

    def predict(self, X):
        """ Returns the prediction of the MLE for the given data.
        n and m are the number of input and output units, respectively,
        and N is the number of samples.
        Args:
            - X (n by N array): sample features
        """
        X_n = (X - self.data_mean)/self.data_std
        return self.W_2.T @ expit(self.W_1.T @ X_n + self.b_1)