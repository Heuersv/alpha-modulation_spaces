import numpy as np
import multiprocessing as mp

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

alphas = [0]
js = range(-10, 10)
ks = range(-10, 10)
result = np.zeros((len(alphas), len(ks), len(js)))


def get_coefficient(alpha_ind: int, j_ind: int, k_ind: int):
    alpha1 = alphas[alpha_ind]
    j1 = js[j_ind]
    k1 = ks[k_ind]

    result[alpha_ind, k_ind, j_ind] = unweighted_coefficient(signal, window, j1, k1, epsilon, alpha1)


pool = mp.Pool(processes=8)
results_after_processing = [pool.apply_async(get_coefficient, (alpha_index, j_index, k_index))
                            for alpha_index in range(len(alphas))
                            for j_index in range(len(js))
                            for k_index in range(len(ks))]


plot_matrix(result[0, :, :])
