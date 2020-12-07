import numpy as np

from short_variables import x_jk, omega_j, beta_alpha
from operators import translation, modulation, dilation
from calculations import scalar_product


# The atom psi_{j,k} for the scalar product with the signal
def alpha_modulation_atom(psi, j, k, epsilon, alpha):
    translation_parameter = x_jk(j, k, epsilon, alpha)
    modulation_parameter = omega_j(epsilon, j, alpha)
    dilation_parameter = beta_alpha(modulation_parameter, alpha)

    psi = translation(psi, translation_parameter)
    psi = modulation(psi, modulation_parameter)
    psi = dilation(psi, dilation_parameter)

    return psi


# Get the coefficient, not yet weighted
def unweighted_coefficient(signal, psi, j, k, epsilon, alpha):
    atom = alpha_modulation_atom(psi, j, k, epsilon, alpha)

    return np.abs(scalar_product(signal, atom))


# Every coefficient is weighted differently
def coefficient_weight(j, s, alpha, p):
    exponent = s - alpha * (1/p - 1/2) / (1 - alpha)

    return np.power(1 + (1 - alpha) * np.abs(j), exponent)


# Get the weighted coefficients. Not yet to the power of p.
def weighted_coefficient(signal, psi, j, k, epsilon, s, alpha, p):
    coef = unweighted_coefficient(signal, psi, j, k, epsilon, alpha)
    weight = coefficient_weight(j, s, alpha, p)

    return coef * weight
