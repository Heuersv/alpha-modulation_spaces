import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from plots import plot_matrix, plot_callable
from functions import doppler_signal, b_spline
from coefficients import unweighted_coefficient, weighted_coefficient

modulation_smoothness = 1
s = 1
epsilon = 0.1
time_location = 40
frequency = 200

signal = doppler_signal(time_location, frequency, interval=[0, 1])
window = b_spline(4, [0, 1])

alphas = [0, 0.5, 0.9]
js = range(-10, 10)
ks = range(-10, 10)
result_dict = {}


def get_coefficients(alpha):
    coefficients = np.zeros((len(ks), len(js)))
    for j_index in range(len(js)):
        for k_index in range(len(ks)):
            j1 = js[j_index]
            k1 = ks[k_index]
            coefficients[k_index][j_index] = unweighted_coefficient(signal, window, j1, k1, epsilon, alpha)
    return alpha, coefficients


def log_result(coefs):
    result_dict[coefs[0]] = coefs[1]


if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count() - 1)
    for a in alphas:
        pool.apply_async(get_coefficients, args=(a, ), callback=log_result)
    pool.close()
    pool.join()
    # for alpha in alphas:
    #     get_coefficients(alpha)

    for a in alphas:
        plot_matrix(result_dict[a])
    plt.show()
