"""Volume 2
<Mingyan Zhao>
<Math 323>
<02/14/2019>
"""
import numpy as np
import scipy.linalg as la
from autograd import grad

def bfgs(f, x0, A0_inv, e = 1e-5, M = 20):
    converged = True
    df = grad(f)
    for i in range(M):
        x1 = x0 - A0_inv @ df(x0)
        if la.norm(x1-x0) <e:
            converged = True
            break
        s, y = x1-x0, df(x1).T - df(x0).T
        syi, syo, yso, ss = np.inner(s,y), np.outer(s,y), np.outer(y,s), np.outer(s,s)

        A0_inv = A0_inv + (syi + (y@A0_inv @y)) * ss/(syi)**2 - (A0_inv @ yso + syo@A0_inv)/(syi)
        x0 = x1
    return x1, converged, i+1

def prob28():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    x0 = np.array([-2.,2.])
    A0_inv = np.array([[1.,0.],[0.,1.]])
    return bfgs(f, x0, A0_inv)
