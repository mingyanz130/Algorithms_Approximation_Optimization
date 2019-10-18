"""Volume 2
<Mingyan Zhao>
<Math 323>
<01/17/2019>
"""
import numpy as np
import matplotlib.pyplot as plt

def gauss_quad(f, n):
    points, w = np.polynomial.legendre.leggauss(n+1)
    return f(points).T@w

def prob_45(f):
    domain = 10*np.arange(1,11)
    range = []
    for n in domain:
        range.append(gauss_quad(f, n))
    plt.plot(domain, range)
    plt.show()
