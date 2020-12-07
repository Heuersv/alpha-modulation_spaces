# All these operators work on callables and return the result as a callable
import numpy as np


# Translation of function f with parameter x
def translation(f: callable, x):
    if not callable(f):
        raise TypeError('First argument (function) is not callable')

    return lambda t: f(t - x)


# Modulation of function f with parameter omega
def modulation(f: callable, omega):
    if not callable(f):
        raise TypeError('First argument (function) is not callable')

    return lambda t: np.exp(2 * np.pi * 1j * omega * t) * f(t)


# Dilation of function f with parameter a
def dilation(f: callable, a):
    if not callable(f):
        raise TypeError('First argument (function) is not callable')

    return lambda t: np.power(np.abs(a), -1/2) * f(t/a)


# Complex conjugate of a function (calculated element-wise)
def conjugation(f: callable):
    if not callable(f):
        raise TypeError('Argument (function) is not callable')

    return lambda t: np.conjugate(f(t))


# Multiplication of two functions (element-wise)
def multiplication(f: callable, g: callable):
    if not callable(f):
        raise TypeError('First argument (function) is not callable')
    if not callable(g):
        raise TypeError('Second argument (function) is not callable')

    return lambda t: f(t) * g(t)
