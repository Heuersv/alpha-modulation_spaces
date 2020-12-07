import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

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


# Get the coefficients. Uncomment the version you want to use. Also returns the parameter alpha to use for logging
def get_coefficients(alpha):
    coefficients = np.zeros((len(ks), len(js)))
    for j_index in range(len(js)):
        for k_index in range(len(ks)):
            j1 = js[j_index]
            k1 = ks[k_index]
            coefficients[k_index][j_index] = unweighted_coefficient(signal, window, j1, k1, epsilon, alpha)
            # coefficients[k_index][j_index] = weighted_coefficient(signal, window, j1, k1, epsilon, s, alpha, p)
    return coefficients, alpha


# Write the results to the result dictionary. Different function to use as callback
result_dict = {}
def log_result(coefficients):
    result_dict[coefficients[1]] = coefficients[0]


if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count() - 1)
    for a in alphas:
        pool.apply_async(get_coefficients, args=(a,), callback=log_result)
    pool.close()
    pool.join()

    for a in alphas:
        plot_matrix(result_dict[a])
    plt.show()
