"""Volume 2
<Mingyan Zhao>
<Math 323>
<02/07/2019>
"""
import numpy as np
import numpy.linalg as la
from autograd import grad, jacobian
from autograd import numpy as anp
from scipy.optimize import rosen



def prob19(f, x0, tol=1e-5, maxiter=100):


    df = grad(f)
    d2f = jacobian(df)


    #repeat maxiter times
    for k in range(maxiter):
        #update points

        x1 = x0 - la.solve(d2f(x0), df(x0))

        #check if it converged
        if la.norm(x1 - x0) < tol:
            return x1, True, k+1
        x0 = x1

    return x1, False, maxiter

def prob20():
    return prob19(rosen, np.array([-2.0,2.0]))

def prob22(r, x0, tol=1e-5, maxiter=100):

    J = jacobian(r)
    #repeat maxiter times
    for k in range(maxiter):
        #update points

        x1 = x0 - la.solve(J(x0).T@J(X0), J(x0).T@r(X0))

        #check if it converged
        if la.norm(x1 - x0) < tol:
            return x1, True, k+1
        x0 = x1

    return x1, False, maxiter

def rangefinder():
    d = anp.array([3.88506517,2.87540403,3.10537735,3.99674185])
    a = anp.array([[0.,0.],[1.0,1.0],[2.,0.],[-1.,3.]])
    r = lambda x: d-anp.linalg.norm(x-a, axis=1)
    x0 = anp.array([2.,2.])
    return probb22(r,x0)
