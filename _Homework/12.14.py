"""Volume 2
<Mingyan Zhao>
<Math 323>
<02/07/2019>
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import time
import sympy as sy
from autograd import grad
from autograd import elementwise_grad

def prob_14(Q, b, x0, ϵ=1e-15):
    df =lambda x: Q@x - b
    i = 0
    while True:
        ak = df(x0)@df(x0).T/(df(x0)@Q@df(x0).T)
        x1 = (x0-ak*df(x0))
        x0 =x1
        i += 1
        if la.norm(df(x0))<ϵ:
            return x0, i


def prob15(f, x0, ϵ=1e-5):

    df = grad(f)
    x0 = 1.*x0
    phi = lambda a: f(x0-a*df(x0))
    d2f = elementwise_grad(df)
    a0 = 1.
    for i in range(501):
            ak = prob9(phi,a0,ϵ,500)
            #check if it converged
            x1 = (x0-ak*df(x0).T)

            if la.norm(df(x1))<ϵ:
                return x1, i+1
            x0 =x1
    return x1, i

def prob9(f,x0,ϵ,M):
    conv = False
    df = elementwise_grad(f)
    d2f = elementwise_grad(df)
    for i in range(M):
        x1 = (x0-df(x0)/d2f(x0))
        if abs(x1-x0) < ϵ:
            conv = True
            break
        x0 =x1
    return x1

#prob16()
#It dose converge, it took 317 iterations.
