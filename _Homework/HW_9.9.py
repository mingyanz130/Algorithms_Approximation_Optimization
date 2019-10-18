"""Volume 2:
<Mingyan Zhao>
<Math 322>
<01/09/2019>
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fac
from scipy.special import comb

#problem 9.9

def weights(list1):
    #turn the input into a numpy array
    list1 = np.array(list(list1))
    n = len(list1)
    W = np.zeros(n)
    for i in range(n):
        list_ = (list1[i]-list1)
        list_ = np.delete(list_, i)
        p = 1
        for k in list_:
            p *= k
        W[i] = p
    return 1/W

def bary(A, x):
    #input is a matrix of coordinates
    x_ = A[:,0]
    y_ = A[:,1]
    W = weights(x_)
    n = A.shape[0]
    num = 0
    den = 0
    for i in range(n):
        num += W[i]*y_[i]/(x-x_[i])
        den += W[i]/(x-x_[i])
    return num/den

def graph(n):
    f = lambda x: abs(x )
    x = np.linspace(-5,5,200)
    xn = np.linspace(-5,5,n)
    A = np.zeros((n,2))
    A[:,0] = xn
    A[:,1] = f(xn)
    plt.plot(x,bary(A,x), label = "Interporlation")
    plt.plot(x, f(x), label = "Original")
    plt.title("n = " + str(n))
    plt.legend()

def graphall():
    for n in range(2,21):
        plt.subplot(4,5,n-1)
        graph(n)
    plt.show()
