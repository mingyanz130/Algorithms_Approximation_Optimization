"""Volume 2:
<Mingyan Zhao>
<Math 321>
<12/06/2018>
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def phi(t,j,k):
    if (t < (k+1)/(2**j) and t >= k/(2**j)):
        return 1
    else:
        return 0

def psi(t,j,k):
    return phi(j+1,2*k,t) - phi(j+1,2*k+1,t)



def FWT(wavelet, j=0):
    a = np.array(wavelet)
    m = int(np.log2(len(a)))
    b = []
    while m > j:
        b.append(0.5 * (a[::2]-a[1::2]))
        a = 0.5 * (a[::2] + a[1::2])
        m -= 1

    b = b[::-1]
    c = b[0]
    for i in range(1,len(b)):
        c = np.concatenate([c,b[i]] )
    return np.concatenate([a,c])


def prob38(wavelet, j):
    def f(t, wavelet):
        f = phi(t,0,0)*wavelet[0]
        index =1
        for i in range(j):
            for k in range(2**i):
                 f += psi(t,i,k)*wavelet[index]
                 index += 1
        return f
    def Tj(t,j,wavelet):
        k = np.floor((2**j)*t)
        return f(k/2**j, wavelet)
    return lambda t: Tj(t,j, wavelet)

def plot38(j):
    f = lambda t: np.ceil(100*t**2*(1-t)*abs(np.sin(10*t/3)))
    d = np.linspace(0,1,2**8, endpoint = False)

    k = np.arange(2**j)
    a_k = f(k/2**j)
    B = FWT(a_k)
    Tj = prob38(B,j)

    plt.subplot(121)
    plt.plot(d, f(d))
    plt.title("f(t)")
    plt.subplot(122)
    tj = []
    for i in d:
        tj.append(Tj(i))
    plt.plot(d,tj)
    plt.title("Estimate j = " + str(j))

    plt.show()

def prob39(f, l, j):
    return prob40(f,l,j,0,1)

def plot39():
    f = lambda x: np.sin(2*np.pi*x-5)/np.sqrt(abs(x-np.pi/20))
    l, a, b = 10, -1, 1
    d = np.linspace(0,1,2048)
    for j in range(l):
        Tj = prob39(f,l,j)
        plt.subplot(2,5,j+1)
        plt.title("Tj[f], j = " + str(j))
        plt.plot(d, f(d))
        tj = []
        for i in d:
            tj.append(Tj(i))
        plt.plot(d,tj)
        plt.title("Estimate j = " + str(j))
    plt.tight_layout()
    plt.show()

def prob40(f, l, j, a, b):
    k = np.linspace(a,b,2**l, endpoint = False)
    a_k = f(k)
    B = FWT(a_k)
    return prob38(B, j)

def plot40():
    f = lambda x: np.sin(2*np.pi*x-5)/np.sqrt(abs(x-np.pi/20))
    l, a, b = 10, -1, 1
    d = np.linspace(0,1,2048)
    for j in range(l):
        Tj = prob40(f,l,j,a,b)
        plt.subplot(2,5,j+1)
        plt.title("Tj[f], j = " + str(j))
        plt.plot(d, f(d))
        tj = []
        for i in d:
            tj.append(Tj(i))
        plt.plot(2*d-1,tj)
        plt.xlim(-1,1)
        plt.title("Estimate j = " + str(j))
    plt.tight_layout()
    plt.show()
