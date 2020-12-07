import numpy as np
from scipy.interpolate import BSpline


# B-Spline as a callable function
def b_spline(order, interval=None):
    if interval is None:
        interval = [0, 1]
    x = np.linspace(interval[0], interval[1], order + 1)
    result = BSpline.basis_element(x, extrapolate=False)

    return lambda t: float(result(t)) if interval[0] <= t <= interval[1] else 0


# A possible signal (the synthetic signal from our paper) as a callable function
def doppler_signal(time_location, frequency, interval=None):
    if interval is None:
        interval = [0, 1]

    result = lambda t: np.sin(2 * np.pi * frequency * t * np.exp(-time_location * np.power(t - np.mean(interval), 2)))

    return lambda t: result(t) if interval[0] <= t <= interval[1] else 0
