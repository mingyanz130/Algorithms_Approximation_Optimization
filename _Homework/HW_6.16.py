"""HW_6.16
<Mingyan Zhao>
<Math 321>
<11/09/2018>
"""
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import math


def graph():
    """
    beta
    mean = .5
    variance = 1/8

    Uniform
    mean = .5
    variance = 1/12
    """
    #plotting
    
    x = np.linspace(0,1,100)
    for c, n in enumerate([1,2,4,8,16,32]):
        #Beta
        mean = .5
        variance = 1/8
        plt.subplot(2,6,c+1)
        plt.plot(x, stats.beta.pdf(x, .5, .5), label = "Beta")
        plt.plot(x, stats.norm.pdf(x,mean, math.sqrt(variance/n)), label = "Norm")
        X = [sum(stats.beta.rvs(.5,.5, size = n))/n for i in range(1000)]
        plt.hist(X, density = True, label = "sample means")
        plt.title("Beta n= "+ str(n))
        if c == 0:
            plt.legend()
        #uniform
        mean = .5
        variance = 1/12
        plt.subplot(2,6,c+7)
        plt.plot(x, stats.uniform.pdf(x,0,1), label = "Uniform")
        plt.plot(x, stats.norm.pdf(x,mean, math.sqrt(variance/n)), label = "Norm")
        X = [sum(stats.uniform.rvs(size = n))/n for i in range(1000)]
        plt.hist(X, density = True, label = "sample means")
        plt.title("Uniform n= "+ str(n))
        if c == 0:
            plt.legend()
    plt.suptitle("Plot")
    plt.show()
