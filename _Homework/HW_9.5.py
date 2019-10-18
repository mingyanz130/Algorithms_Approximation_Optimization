"""Volume 2:
<Mingyan Zhao>
<Math 322>
<01/07/2019>
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fac
from scipy.special import comb

def Bernstein(f, n):
    #set the domain
    x = np.linspace(0,1,200)
    #calculate the sum of the bernstein polynomials
    sum = 0
    for k in range(n+1):
        sum += f(k/n)* comb(n,k)*(x**k)*((1-x)**(n-k))
    #plot the bernstein transformation and the original
    plt.plot(x,sum, label = "Bernstein")
    plt.plot(x,f(x), label = "origin")
    plt.title("n = " + str(n))
    plt.legend()
    
#graph for different n
def Bernstein_graph(f):
    list = [4,10,50,200]
    for i in range(4):
        plt.subplot(2,2,i+1)
        Bernstein(f, list[i])
    plt.show()
