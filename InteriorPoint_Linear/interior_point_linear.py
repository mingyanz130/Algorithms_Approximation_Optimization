# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add slack variables.
    Parameters:
        m -- positive integer: the number of desired constraints
             and the dimension of space in which to optimize.
    Returns:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
    """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Returns:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    #initilize the size
    m,n = len(b), len(c)
    #define F
    F = lambda x,lam,mu:np.concatenate([A.T@lam+mu-c, A@x-b, np.diag(mu)@x]).astype(float)

    O1,O2,O3 = np.zeros((n,n)), np.zeros((m,n+m)), np.zeros((n,m))
    I = np.eye(n)
    #define DF
    DF = lambda x, lam, mu: np.block([[O1,A.T,I],[A,O2],[np.diag(mu),O3,np.diag(x)]])

    #find the search direction
    def searchdirection(x,lam,mu):
        sigma =0.1
        v = x.T@mu/n
        measure = v
        v = sigma*v*np.ones(n)
        v = np.concatenate([np.zeros(m+n),v])
        #solve the system of linear approcimation
        lu, piv = la.lu_factor(DF(x,lam,mu))
        return la.lu_solve((lu,piv), -F(x,lam,mu)+v), measure

    def step_length(x,lam,mu):
        #get the search direction
        answer,measure = searchdirection(x,lam,mu)
        x_ = answer[:n]
        lam_ = answer[n:n+m]
        mu_ = answer[n+m:]
        #find the maximum allowable step length
        alpham = min(1,min(-mu[mu_<0]/mu_[mu_<0]))
        deltam = min(1,min(-x[x_<0]/x_[x_<0]))
        alpha = min(1,0.95*alpham)
        delta = min(1,0.95*deltam)
        return x+delta*x_, lam+alpha*lam_, mu + alpha*mu_, measure

    x,lam,mu = startingPoint(A,b,c)
    #loop until condition is met
    for i in range(niter):
        x,lam,mu, measure = step_length(x,lam,mu)
        if measure < tol:
            return x, c.T@x
    return x, c.T@x







def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""

    #load the data
    with open(filename) as file:
        data = file.readlines()

    #load the data as a matrix
    parse = lambda x: x.strip().split()
    data = np.array(list(map(parse,data)),dtype=float)

    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]


    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)


    #solve for the slope and intercept of least abosolute value
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    #solve for the slope and intercept of least square
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0,10,200)

    #scatter plot the original data
    plt.scatter(data[:,1],data[:,0],label="data")
    #least absolute deviations
    plt.plot(domain,beta*domain+b,label="LAD")
    #least square solutions
    plt.plot(domain, domain*slope + intercept,label="Least Square")

    plt.legend()
    plt.title("least absolute deviation and least squares")
    plt.show()

if __name__ == "__main__":
    m,n= 6,3
    A, b, c, x = randomLP2(m,n)
    #print(A,b,c,x)
    point, value = interiorPoint(A, b, c)
    print(x,point[:n])
    print(np.allclose(x, point[:n]))
    #leastAbsoluteDeviations(filename='simdata.txt')
