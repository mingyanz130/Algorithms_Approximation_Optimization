"""Volume 2
<Mingyan Zhao>
<Math 323>
<01/17/2019>
"""
import numpy as np
import matplotlib.pyplot as plt

def prob5():
    f = lambda x: (1-x)-1
    g = lambda x: -x
    h = lambda x: ((1-x)-1)/x

    domain = np.linspace(-3*10**(-15), 3*10**(-15),1000)
    plt.subplot(121)
    plt.plot(domain,f(domain))
    plt.plot(domain,g(domain))
    plt.subplot(122)
    plt.plot(domain,h(domain))

    plt.show()

def prob():
    check = 1

    x0 = 0.0001
    x1 = x0 - x0**3/2
    while x1 > 0.00005:
        x1 = x0 - x0**3/2
        x0 = x1
        check += 1

    return check
