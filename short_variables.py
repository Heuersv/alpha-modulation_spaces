# Some variable definitions to match the paper
import numpy as np


def beta_alpha(omega, alpha):
    if alpha < 0 or alpha >= 1:
        raise ValueError('Use alpha in [0,1)')

    return np.power(1 + np.abs(omega), -alpha)


def p_alpha(omega, alpha):
    if alpha < 0 or alpha >= 1:
        raise ValueError('Use alpha in [0,1)')

    return np.sign(omega) * (np.power(1 + (1 - alpha) * np.abs(omega), 1 / (1 - alpha)) - 1)


def omega_j(epsilon, j, alpha):
    if alpha < 0 or alpha >= 1:
        raise ValueError('Use alpha in [0,1)')
    if epsilon <= 0:
        raise ValueError('Use epsilon > 0')

    return p_alpha(epsilon * j, alpha)


def x_jk(j, k, epsilon, alpha):
    if alpha < 0 or alpha >= 1:
        raise ValueError('Use alpha in [0,1)')
    if epsilon <= 0:
        raise ValueError('Use epsilon > 0')
    if not isinstance(j, int):
        raise TypeError('j not an integer')
    if not isinstance(k, int):
        raise TypeError('k not an integer')

    return epsilon * beta_alpha(omega_j(epsilon, j, alpha), alpha) * k
