import numpy as np
from scipy.integrate import quad

from operators import conjugation, multiplication


# Scalar product of functions f and g
def scalar_product(f, g):
    integrand = multiplication(f, conjugation(g))

    return quad(integrand, -np.inf, np.inf)[0]
