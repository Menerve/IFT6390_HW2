import numpy as np
from kernels import *

class SVMKernel:
    def __init__(self, mu, c_errors, batch, kernel_function, k=2, sigma=1):
        self.batch = batch
        self.c_errors = c_errors
        self.mu = mu
        self.kernel_function = kernel_function
        self.k = k
        self.sigma = sigma

    def alphas(self, x, xj):
        alphas = 0
        for i in range(x.shape[0]):
            if self.kernel_function == scalar_kernel:
                alphas += self.alpha[i] * self.kernel_function(x[i], xj)
            elif self.kernel_function == polynomial_kernel:
                alphas += self.alpha[i] * self.kernel_function(x[i], xj, self.k)
            else:
                alphas += self.alpha[i] * self.kernel_function(x[i], xj, self.sigma)
        return alphas

    def train(self, train_data, epochs, epochs_to_display=[], display_errors=False):
        self.alpha = np.zeros(train_data.shape[0])
        self.biais = 0
        n_examples = train_data.shape[0]
        data = train_data[:, :-1]
        self.data = train_data
        targets = train_data[:, -1]

        for j in range(epochs):
            i = j % n_examples

            if self.alphas(data, data[i]) + self.biais * targets[i] >= 1:
                self.alpha *= 1 - self.mu / (n_examples ** 2 * self.c_errors)
            else:
                self.alpha *= 1 - self.mu / (n_examples ** 2 * self.c_errors)
                self.alpha[i] = self.alpha[i] + self.mu / (n_examples * targets[i])
                self.biais = self.biais + self.mu * targets[i] / n_examples

    def compute_predictions(self, test_data):
        sorties = []

        for i in range(len(test_data)):
            data = []
            for j in range(len(test_data[i])):
                data.append(test_data[i][j])
            data.append(1)
            sorties.append(np.sign(self.alphas(self.data, data) + self.biais))

        return sorties