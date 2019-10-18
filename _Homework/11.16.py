"""Volume 2
<Mingyan Zhao>
<Math 323>
<01/17/2019>
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import time
import sympy as sy
from autograd import grad
from autograd import numpy as anp
from autograd import elementwise_grad

f1= lambda p: p[0]/(p[0]+2+p[1]**2)

def prob16i(f=f1,n=2,x=[2,3],h = 2*np.sqrt(1.11e-16)):

    #initialize the Jacobian matrix
    J = np.zeros((1,n))
    #get the identity matrix
    I = np.eye(n)
    #Approximate the Jacobian matrix of f at x using the first order
    #forward difference quotient
    for j in range(n):
        J[:,j] = (f(x+h*I[:,j])-f(x))/h
    return J


def prob16iii(f=f1):
    domain = (1/2)**np.arange(2,54)
    answer = []
    for h in domain:
        answer.append(prob16i(f,2,[2,3],h))
    return answer

def prob16iv(f=f1):
    domain = (1/2)**np.arange(2,54)
    Ti = []
    for h in domain:
        start = time.time()
        prob16i(f,2,[2,3],h)
        Ti.append(time.time() - start)
    answer = prob16iii()
    for k in range(len(answer)):
        answer[k] = la.norm(answer[k]-np.array([11/169,-12/169]))
    plt.loglog(domain,Ti, basex = 2)
    plt.loglog(domain,answer, basex = 2)
    plt.show()
    return domain[np.argmin(answer)]


def prob17i(f=f1,n=2,x=[2,3],h = 1.4*(1.11e-16)**(1/3)):

    #initialize the Jacobian matrix
    J = np.zeros((1,n))
    #get the identity matrix
    I = np.eye(n)
    #Approximate the Jacobian matrix of f at x using the second order
    #centered difference quotient
    for j in range(n):
        J[:,j] = (f(x+h*I[:,j])-f(x-h*I[:,j]))/(2*h)
    return J

def prob17ii(f=f1):
    domain = (1/2)**np.arange(2,54)
    answer = []
    for h in domain:
        answer.append(prob17i(f,2,[2,3],h))
    return answer

def prob17iii(f=f1):
    domain = (1/2)**np.arange(2,54)
    Ti = []
    for h in domain:
        start = time.time()
        prob16i(f,2,[2,3],h)
        Ti.append(time.time() - start)

    answer = prob17ii()
    for k in range(len(answer)):
        answer[k] = la.norm(answer[k]-np.array([11/169,-12/169]))
    plt.loglog(domain,Ti, basex = 2)
    plt.loglog(domain,answer, basex = 2)
    plt.show()
    return domain[np.argmin(answer)]
#the results from different methods:
#forward:  h = 2.9802322387695312e-08
#centered:  h = 7.62939453125e-06



f2= lambda x: (anp.sin(x)**3+anp.cos(x))/np.e**x



#problem 18
def symbol(x0=1.5):
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    #initialize the function and find its derivative
    x = sy.symbols("x")
    f = (sy.sin(x)**3 + sy.cos(x))/sy.E**x
    df = sy.lambdify(x, sy.diff(f,x))

    return df(x0)

def fdq1(f=f2, x=1.5, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    #the first order forward difference
    f1 = (f(x+h) - f(x))/h
    return f1

def cdq2(f=f2, x=1.5, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    f_ = (f(x+h) - f(x-h))/(2*h)
    return f_

def comp(f=f2, x=1.5, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    f3 = (f(x+1j*h).imag)/h
    return f3

def system(f=f2, x=1.5):
    return grad(f)(x)

def prob18():

    domain = (1/2)**np.arange(2,54)
    time1 = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []
    error1 = []
    error2 = []
    error3 = []
    error4 = []
    error5 = []
    rate1 = []
    rate2 = []
    rate3 = []
    rate4 = []
    rate5 = []
    for h in domain:
        start = time.time()
        x1 = symbol()
        time1.append(time.time()-start)
        error1.append(0)

        start = time.time()
        x2 = fdq1(h=h)
        time2.append(time.time()-start)
        error2.append(abs(x2-x1))

        start = time.time()
        x3 = cdq2(h=h)
        time3.append(time.time()-start)
        error3.append(abs(x3-x1))

        start = time.time()
        x4 = comp(h=h)
        time4.append(time.time()-start)
        error4.append(abs(x4-x1))

        start = time.time()
        x5 = system()
        time5.append(time.time()-start)
        error5.append(abs(x5-x1))
    plt.subplot(121)
    plt.title("computation time")
    plt.loglog(domain,time1, label = "symbolically", basex = 2)
    plt.loglog(domain,time2, label = "forward", basex = 2)
    plt.loglog(domain,time3, label = "centered", basex = 2)
    plt.loglog(domain,time4, label = "complex", basex = 2)
    plt.loglog(domain,time5, label = "algorithm", basex = 2)
    plt.legend()
    plt.subplot(122)
    plt.title("overall accuracy")
    plt.loglog(domain,error1, label = "symbolically", basex = 2)
    plt.loglog(domain,error2, label = "forward", basex = 2)
    plt.loglog(domain,error3, label = "centered", basex = 2)
    plt.loglog(domain,error4, label = "complex", basex = 2)
    plt.loglog(domain,error5, label = "algorithm", basex = 2)
    plt.legend()
    plt.show()



def prob19():
    return None
