#Alassane Ndiaye
#David Krueger
#Thomas Rohee

import numpy as np


def scalar_kernel(x_i, x_y):
    return np.dot(x_i, x_y)


def polynomial_kernel(x_i, x_y, k=2):
    return (1 + np.dot(x_i, x_y))**k


def rbf_kernel(x_i, x_y, sigma=0.2):
    return np.exp(-1/2*(np.linalg.norm(x_i - x_y)**2)/sigma**2)