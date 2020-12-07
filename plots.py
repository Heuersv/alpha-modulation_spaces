import numpy as np
import matplotlib.pyplot as plt


# Plot a matrix for e.g. spectrogram visualization
def plot_matrix(matrix):
    plt.figure()
    plt.imshow(matrix)
    # plt.show()


# Plot a callable function on an interval
def plot_callable(f, interval, samples=100):
    result = []
    sampled_interval = np.linspace(interval[0], interval[1], samples)
    for t in sampled_interval:
        result.append(f(t))
    plt.figure()
    plt.plot(sampled_interval, result)
    plt.show()
