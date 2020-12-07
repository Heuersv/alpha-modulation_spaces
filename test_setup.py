import numpy as np

from plots import plot_matrix, plot_callable
from functions import doppler_signal, b_spline
from coefficients import unweighted_coefficient, weighted_coefficient


# Parameters for Modulation space
modulation_smoothness = 1
s = 1
epsilon = 0.1

# Parameters for signal and window
time_location = 40
frequency = 200
spline_order = 4
interval_signal = [0, 1]
interval_spline = [0, 1]
signal = doppler_signal(time_location, frequency, interval=interval_signal)
window = b_spline(spline_order, interval=interval_spline)

# Parameters for the transform
alphas = [0, 0.5, 0.9]
js = range(-10, 10)
ks = range(-10, 10)

# 3 dimensional result matrix
result = np.zeros((len(alphas), len(ks), len(js)))

# Calculation
for alpha_index in range(len(alphas)):
    for j_index in range(len(js)):
        for k_index in range(len(ks)):
            alpha = alphas[alpha_index]
            j = js[j_index]
            k = ks[k_index]

            result[alpha_index, k_index, j_index] = unweighted_coefficient(signal, window, j, k, epsilon, alpha)
            # result[alpha_index, k_index, j_index] = weighted_coefficient(signal, window, j, k, epsilon, s, alpha, p)


# Plot the result for the first alpha
plot_matrix(result[0, :, :])
