"""Volume 2:
<Mingyan Zhao>
<Math 322>
<01/09/2019>
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fac
from scipy.special import comb

#problem 9.17-18

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
#problem 25
def prob25(xn= [-1,-1/3,1/3,1]):
    """
    input is the x_coordinates
    """
    f = lambda x: np.sin(np.pi*x)
    x = np.linspace(-1,1,200)
    xn = np.array(xn)
    A = np.zeros((len(xn),2))
    A[:,0] = xn
    A[:,1] = f(xn)
    plt.plot(x,bary(A,x), label = "Interporlation")
    plt.plot(x, f(x), label = "Original")
    plt.legend()
    plt.show()

#problem 26
def prob26():
    """
    input is the x_coordinates
    """
    xn= [-1,-1/3,1/3,1]
    xn2 = [np.cos(j*np.pi/3) for j in range(4)]
    f = lambda x: np.sin(np.pi*x)
    x = np.linspace(-1,1,200)
    xn = np.array(xn)
    xn2 = np.array(xn2)
    A = np.zeros((len(xn),2))
    A[:,0] = xn
    A[:,1] = f(xn)
    A2 = np.zeros((len(xn2),2))
    A2[:,0] = xn2
    A2[:,1] = f(xn2)
    plt.plot(x,bary(A2,x), label = "Chebyshev")
    plt.plot(x,bary(A,x), label = "Interporlation")
    plt.plot(x, f(x), label = "Original")
    plt.legend()
    plt.show()

#problem 28
def prob28():
    domain = np.linspace(1,20,500)
    xn = np.array([np.cos((j+1/2)*np.pi/20) for j in range(20)])
    xn = (20-1)*xn/2+(20+1)/2
    xn2 = np.array([i for i in range(1,21)])
    prod1 = 1
    prod2 = 1
    for i in range(20):
        prod1 *= domain - xn[i]
        prod2 *= domain - xn2[i]
    plt.plot(domain,prod1,label="Chebyshev")
    plt.plot(domain,prod2, label="Wilkinson")
    plt.legend()
    plt.title("sup(q) = "+str(max(prod1))+"  sup(W)= "+str(max(prod2)))
    plt.show()



def graphall():
    for n in range(2,21):
        plt.subplot(4,5,n-1)
        graph(n)
    plt.show()
