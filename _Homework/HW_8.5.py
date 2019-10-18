"""Volume 2: hw 8.25
<Mingyan Zhao>
<Math 321>
<11/19/2018>
"""
import matplotlib.pyplot as plt
import numpy as np
import random

#8.25

def prob1():

    """
    let f(x) = x and g(x) = sin(x)
    Sample both functions 1000 times
    on the interval[0,2pi] to get vector f, and g
    computer the convolution f*g and the Hadamard product f.g
    """
    #calculate f, g, fg convolution, and fg Hadamard product
    f = np.linspace(0,2*np.pi,1000)
    g = np.sin(f)
    fgcon = np.convolve(np.hstack([f,f]),g, mode ='valid')[1:]
    fgma = np.multiply(f,g)
    #plot each graph
    plt.subplot(231)
    plt.title("f")
    plt.plot(f, f)
    plt.subplot(232)
    plt.title("g")
    plt.plot(f, g)
    plt.subplot(233)
    plt.title("convolution")
    plt.plot(f, fgcon)
    plt.subplot(234)
    plt.title("Hadamard")
    plt.plot(f, fgma)
    plt.subplot(235)
    plt.title("All together")
    plt.plot(f, f)
    plt.plot(f, g)
    plt.plot(f, fgcon)
    plt.plot(f, fgma)
    plt.show()
