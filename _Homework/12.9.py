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



def prob9(f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    df = grad(f)
    d2f = grad(df)
    #repeat maxiter times
    for k in range(maxiter):
        #update points
        x1 = x0 - df(x0)/d2f(x0)
        #check if it converged
        if abs(x1 - x0) < tol:
            return x1, True, k+1
        x0 = x1

    return x1, False, maxiter

def prob10(b,x, x0 = 0.1, x1 =0.5,tol=1e-15, maxiter=100):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    f = lambda y: b**y - x
    #repeat maxiter times
    for k in range(maxiter):
        x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        #check if it converged
        if abs(x2 - x1) < tol:
            return x2, True, k+1
        x0 = x1
        x1 = x2

    return x2, False, maxiter
