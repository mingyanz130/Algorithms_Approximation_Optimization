"""Volume 2: hw_7.1-7,5
<Mingyan Zhao>
<Math 321>
<11/19/2018>
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
import scipy

def prob6():
    n = 10**5
    #part 1
    #draw samples and only take the ones that is great than 3
    X1 = np.random.normal(0,1,n)
    mask = X1 >=3
    X1 = X1[mask]
    estimate1 = len(X1)*np.sqrt(2*np.pi)/n
    SE1 = np.sqrt(np.sum((np.sqrt(2*np.pi)-estimate1)**2)/(n*(n-1)))

    #part2
    X2 = np.random.normal(3,1,n)
    mask = X2 >=3
    X2 = X2[mask]
    f1 = np.exp(-X2**2/2)
    f2 = 1/np.sqrt(2*np.pi)*np.exp(-(X2-3)**2/2)
    estimate2 = np.sum(f1/f2)/n
    SE2 = np.sqrt(np.sum((f1/f2-np.mean(f1/f2))**2)/(n*(n-1)))


    return "The estimate and error with norm(0,1) is " + str(estimate1) + \
    " and " + str(SE1) + " The estimate and error with norm(3,1) is " + str(estimate2) + " and "+ str(SE2)

def prob7(a=1,b=1,n=10**7):

    x = stats.beta(a,b).rvs(size=n)
    x1 = x*2*np.pi
    f = 2*np.pi/(x1**3+x1+1)
    bf = stats.beta(a, b).pdf(x)
    mean = np.sum(f/bf)/n
    SE = np.sqrt(np.sum((f/bf-np.mean(f/bf))**2)/(n*(n-1)))
    return mean, SE, a, b, n


#7.8
"""
i)finding inverse
solve for x
y = 1-e^(-lambda*x)
e^(-lambda*x) = 1-y
-lambda*x = ln(1-y)
x = F^-1(y) =  -*ln(1-y)/lambda
Thus, it is true.

ii)
P(1-Y<=y) = P(Y<=y)  = P(F(x)<=y) = P(x <= F^-1(y)) = F(F^-1(y)) = y
"""

def prob8():
    n = 10**5
    d = np.linspace(0,1,100)
    u = np.random.uniform(0,1,size = n)
    U = -np.log(u)/2
    plt.hist(U, bins = d, density = True)
    y = stats.gamma.pdf(d, a = 1, scale=-.5)
    plt.plot(d, y)
    plt.title("Problem 8 iii")
    plt.show()

#7.9
"""
i)solve for x
y = 1/(1+e^-x)
1+e^-x = 1/y
e^-x = 1/y-1
-x = ln(1/y-1)
x = F-1(y) = ln(y/(1-y))
"""

def prob9():
    n = 10**5
    d = np.linspace(-10,10,100)
    u = np.random.uniform(0,1,n)
    U = -np.log(1/u-1)
    plt.hist(U, bins = d, density = True)
    plt.plot(d, stats.logistic.pdf(d))
    plt.title("Problem 9 ii")
    plt.show()
    return "the mean is " + str(np.mean(U)) + "and the variance is " + str(np.var(U))

def prob10(d):
    """
    estimates of the volume of the unite ball in d-dimension space
    """

    n= 10**5
    U = np.random.uniform(0,1,(d,n))**2
    mean = np.sqrt(np.sum(U, axis = 0))
    estimation= np.sum(mean <= 1)/n
    return "for d = " + str(d) +", the estimates is " + str(estimation)

#7.11
"""
i)e^(-x^2-x^3) <= me^(-x)
e^(-x^2-x^3)/e^(-x) <=m
e^(-x^2-x^3+x) <=m
the maximum for -x^2-x^3+x is 5/27, so the smallest m will be e^(5/27)

ii)
Let M = Z/m
F_Q(x) = Gamma(1,1)= e^(-x)
F_P(x) = 1/Z*e^(-x^2-x^3)
F_P(x)/F_Q(x) = (1/Z)*Z*F_P(x)/(M/Z*F_Q(x))= e^(-x^2-x^3)/(m*F_Q(x))

"""
def prob11():
    #we sample from Gamma(1,1)
    z =stats.gamma(1).rvs(size = 10**5)
    u = stats.uniform(0,1).rvs(size = 10**5)
    mask = (np.exp(-(z**2+z**3))/(np.exp(5/27)*np.exp(-z)) >= u)
    n = 10**5 - np.sum(mask)

    #repeat untill we obtain all 10**5 random draws
    while n >0:
        z = z[mask]
        u = u[mask]
        new_z = stats.gamma(1).rvs(size=n)
        new_u = stats.uniform(0,1).rvs(size=n)
        z = np.concatenate((z, new_z))
        u = np.concatenate((u, new_u))
        mask = (np.exp(-(z**2+z**3))/(np.exp(5/27)*np.exp(-z)) >= u)
        n = 10**5 - np.sum(mask)
    x = np.linspace(0,2,100)
    plt.hist(z, density= True, bins = x)
    Z = scipy.integrate.quad(lambda i: np.exp(-i**2 - i**3), 0, np.inf)[0]
    y = 1/Z*np.exp(-x**2-x**3)
    plt.plot(x,y)
    plt.show()
