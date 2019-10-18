# solutions.py
"""Volume 2: One-Dimensional Optimization.
<Mingyan Zhao>
<Math 323>
<01/31/2019>
"""
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt


# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Set the initial minimizer approximation as the interval midpoint
    x0 = (a+b)/2
    gr = (1+np.sqrt(5))/2
    #Iterate only maxiter times at most.
    for i in range(maxiter):
        c = (b-a)/gr
        a_ = b-c
        b_ = a + c
        #Get new boundaries for the search interval.
        if f(a_)<=f(b_):
            b = b_
        else:
            a = a_
        #Set the minimizer approximation as the interval midpoint.
        x1 = (a+b)/2
        if abs(x0-x1) < tol:
            #Stop iterating if the approximation stops changing enough.
            return x1, True, i+1
        x0 = x1
    return x1, False, maxiter

# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
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
    #repeat maxiter times
    for k in range(maxiter):
        #update points
        x1 = x0 - df(x0)/d2f(x0)
        #check if it converged
        if abs(x1 - x0) < tol:
            return x1, True, k+1
        x0 = x1

    return x1, False, maxiter
# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
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
    #repeat maxiter times
    for k in range(maxiter):
        #update points
        f1 = df(x1)
        f0 = df(x0)
        x2 = (x0*f1-x1*f0)/(f1-f0)
        #check if it converged
        if abs(x2 - x1) < tol:
            return x2, True, k+1
        x0 = x1
        x1 = x2

    return x2, False, maxiter
# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    #Compute these values only once
    Dfp = Df(x).T@p
    fx = f(x)
    #check the condition
    while f(x+alpha*p) > fx + c*alpha*Dfp:
        alpha = rho*alpha
    return alpha
