from scipy.integrate import quad

from operators import conjugation, multiplication


# Scalar product of functions f and g. f should be list of [signal, interval]
def scalar_product(f, g):
    integrand = multiplication(f[0], conjugation(g))
    interval = f[1]

    return quad(integrand, interval[0], interval[1])[0]
