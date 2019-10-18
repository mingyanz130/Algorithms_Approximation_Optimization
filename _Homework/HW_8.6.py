"""Volume 2:
<Mingyan Zhao>
<Math 321>
<12/06/2018>
"""
from scipy import fftpack
from matplotlib import pyplot as plt
import numpy as np

#8.26
def prob26(n,v,f,T):
    domain = np.arange(n)*T/n
    transform = fftpack.fft(f(domain))/n
    c = np.hstack([transform[n-v:], transform[:v+1]])
    return c

def get_Function(c,v,T):
    return lambda t: np.sum(c[:,np.newaxis]*np.exp(2j*np.pi/T*np.outer(np.arange(-v,v+1),np.array(t))),axis=0).real

g = lambda x: (1-3*np.sin(12*np.pi*x+7) + 5*np.sin(2*np.pi*x-1)+5*np.sin(4*np.pi*x-3))

#8.26
def prob27(n,v,f=g,T=1):
    coeffs = prob26(n,v,f,T)
    new_f = get_Function(coeffs,v, T)

    domain = np.arange(n)*T/n
    t = np.linspace(0,T, 200)

    plt.figure()
    plt.scatter(domain,f(domain))
    plt.plot(t,f(t))
    plt.plot(t,new_f(t))
    plt.title("n="+str(n))
    plt.show()
