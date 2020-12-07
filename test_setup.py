import numpy as np

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

for alpha_index in range(len(alphas)):
    for j_index in range(len(js)):
        for k_index in range(len(ks)):
            alpha = alphas[alpha_index]
            j = js[j_index]
            k = ks[k_index]

            result[alpha_index, k_index, j_index] = unweighted_coefficient(signal, window, j, k, epsilon, alpha)


plot_matrix(result[0, :, :])
