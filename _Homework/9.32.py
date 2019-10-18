"""Volume 2:
<Mingyan Zhao>
<Math 322>
<01/09/2019>
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fac
from scipy.special import comb
from numpy.fft import fft
from numpy.polynomial import chebyshev as cheby

def cheb_interp(f,n):
    y = np.cos((np.pi * np.arange(2*n))/n)
    samples = f(y)

    coeffs = np.real(fft(samples))[:n+1] / n
    coeffs[0] = coeffs[0] / 2
    coeffs[n] = coeffs[n] / 2
    return coeffs


def graph(n):
    f = lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda x: x+1, lambda x: x])
    T = lambda n, x: np.cos(n*np.arccos(x))
    domain = np.linspace(-1,1,200)
    ak = cheb_interp(f,2**n)
    sum = 0
    for i in range(2**n+1):
        sum += ak[i]*T(i,domain)

    plt.plot(domain, f(domain), label = "original")
    plt.plot(domain, sum, label = "chebyshev")
    plt.legend()

def prob32():
    for k in range(1,8):
        plt.subplot(3,3,k)
        graph(k)
        plt.title("n = "+ str(k))
    plt.show()
