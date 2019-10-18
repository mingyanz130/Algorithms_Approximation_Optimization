# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from cvxopt import matrix, solvers


def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    #initilize the size
    m,n = A.shape
    #define F
    F = lambda x,y,mu:np.concatenate([Q@x-A.T@mu+c, A@x-y-b, np.diag(y)@np.diag(mu)@np.ones(m)]).astype(float)

    O1,O2,O3 = np.zeros((m,n)), np.zeros((m,m)), np.zeros((n,m))
    I = np.eye(m)
    #define DF
    DF = lambda x, y, mu: np.block([[Q,O3,-A.T],[A,-I,O2],[O1,np.diag(mu),np.diag(y)]])

    #find the search direction
    def searchdirection(x,y,mu):
        sigma =0.1
        v = y.T@mu/m
        measure = v
        v = sigma*v*np.ones(m)
        v = np.concatenate([np.zeros(m+n),v])
        #solve the system of linear approcimation
        lu, piv = la.lu_factor(DF(x,y,mu))
        return la.lu_solve((lu,piv), -F(x,y,mu)+v), measure

    def step_length(x,y,mu):
        #get the search direction
        answer,measure = searchdirection(x,y,mu)
        x_ = answer[:n]
        y_ = answer[n:n+m]
        mu_ = answer[n+m:]
        #find the maximum allowable step length
        betam = min(1,min(-mu[mu_<0]/mu_[mu_<0]))
        deltam = min(1,min(-y[y_<0]/y_[y_<0]))
        beta = min(1,0.95*betam)
        delta = min(1,0.95*deltam)
        alpha = min(beta,delta)
        return x+alpha*x_, y+alpha*y_, mu + alpha*mu_, measure

    x,y,mu = startingPoint(Q, c, A, b, guess)
    #loop until condition is met
    for i in range(niter):
        x,y,mu, measure = step_length(x,y,mu)
        if measure < tol:
            return x, c.T@x
    return x, c.T@x


def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()


# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """

    # Create the tent pole configuration.
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()
    # Set initial guesses.
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)


    #initilize c, A, H
    H = laplacian(n)
    c = -(n-1)**(-2)*np.ones(n**2)
    A = np.eye(n**2)

    # Calculate the solution.
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))
    # Plot the solution.
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z,  rstride=1, cstride=1, color='r')
    plt.title("optimal elastic tent")
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    #load the data
    with open(filename,'r') as f:
        l = [[float(num) for num in line.split(' ')] for line in f]
    l =np.array(l)
    mu = np.average(l[:,1:],axis=0)
    cov = np.cov(l[:,1:].T)
    
    #set up the matrices
    solvers.options['show_progress'] = False
    Q = matrix(cov)
    p = matrix(np.zeros(8))
    G = matrix(-np.eye(8))
    h = matrix(np.zeros(8))
    A = matrix(np.vstack((np.ones(8),mu)))
    b = matrix([1,1.13])
    #with short selling
    sol1 = solvers.qp(P = Q, q = p, A=A, b=b)
    #without short selling
    sol = solvers.qp(Q, p, G, h, A, b)
    
    return np.ravel(sol1['x']), np.ravel(sol['x'])
    
if __name__ == "__main__":
    """
    Q = np.array([[1,-1],[-1,2]])
    c = np.array([-2,-6])
    A = np.array([[-1,-1],[1,-2],[-2,-1],[1,0],[0,1]])
    b= np.array([-2,-2,-3,0,0])
    guess = np.array([[.5,.5],np.ones(5),np.ones(5)])
    point, value = qInteriorPoint(Q,c,A,b,guess)
    print(point)
    """
    circus(30)
    #print(portfolio(filename="portfolio.txt"))
