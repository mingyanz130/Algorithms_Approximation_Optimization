# cvxopt_intro.py
"""Volume 2: Intro to CVXOPT.
<Name>
<Class>
<Date>
"""
from cvxopt import matrix, solvers
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + 10y + 3z   >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #initilize the matrices
    solvers.options['show_progress'] = False
    c = np.array([2., 1., 3.],dtype="float")
    G = np.array([[-1.,-2.,0],[-2., -10., -3.],[-1.,0.,0.],[0.,-1.,0.], [0., 0.,-1.]],dtype="float")
    h = np.array([-3., -10., 0., 0., 0.],dtype="float")

    #convert the matrices
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    #solve the matrices
    sol = solvers.lp(c, G, h)
    return np.ravel(sol['x']), sol['primal objective']



# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray), without any slack variables u
        The optimal value (sol['primal objective'])
    """
    #set up the matrices
    m,n = A.shape
    solvers.options['show_progress'] = False
    c = np.concatenate([np.ones(n),np.zeros(n)]).astype(float)
    G = np.vstack((np.hstack((-np.eye(n),np.eye(n))), np.hstack((-np.eye(n),-np.eye(n))),np.hstack((-np.eye(n),np.zeros((n,n))))))
    h = np.zeros(3*n).astype(float)
    A = np.hstack((np.zeros((m,n)),A)).astype(float)
    #convert the matrices
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b.astype(float))
    #solve the matrices
    sol = solvers.lp(c, G, h,A,b)

    return np.ravel(sol['x'][n:]),sol['primal objective']

# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #set up the matrices
    solvers.options['show_progress'] = False
    c = np.array([4., 7., 6., 8., 8., 9.])

    G = np.array([[1.,1.,0.,0.,0.,0.],
                [-1.,-1.,0.,0.,0.,0.],
                [0.,0.,1.,1.,0.,0.],
                [0.,0.,-1.,-1.,0.,0.],
                [0.,0.,0.,0.,1.,1.],
                [0.,0.,0.,0.,-1.,-1.],
                [1.,0.,1.,0.,1.,0.],
                [-1.,0.,-1.,0.,-1.,0.],
                [0.,1.,0.,1.,0.,1.],
                [0.,-1.,0.,-1.,0.,-1.]])
    G = np.vstack((G, -1*np.eye(6)))
    h = np.array([7,-7,2,-2,4,-4,5,-5,8,-8,0,0,0,0,0,0],dtype="float")

    #convert the matrices
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    #solve the matrices
    sol = solvers.lp(c, G, h)
    return np.ravel(sol['x']), sol['primal objective']

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #set up the matrices
    solvers.options['show_progress'] = False
    Q = matrix(np.array([[3., 2.,1.],[2.,4.,2.],[1., 2., 3.]]))
    r = matrix([3.,0., 1.])
    #solve the matrices
    sol=solvers.qp(Q, r)
    return np.ravel(sol['x']), sol['primal objective']

# Problem 5
def l2Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_2
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #set up the matrices
    solvers.options['show_progress'] = False
    m,n = A.shape
    Q = matrix(2*np.eye(n))
    r = matrix(np.zeros(n))
    A = matrix(A.astype(float))
    b = matrix(b.astype(float))
    #solve the matrices
    sol=solvers.qp(Q, r,A=A,b=b)
    return np.ravel(sol['x']), sol['primal objective']


# Problem 6
def prob6():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective']*-1000)
    """
    #set up the matrices
    data = np.load("ForestData.npy")
    solvers.options['show_progress'] = False
    data[:,4] *= 1000
    data[:,5] *= 1000
    data[:,6] /= 788

    c = -data[:,3]
    G = -np.block([[data[:,4:7].T],[np.eye(21)]])
    h = -np.concatenate([np.array([40000.,5.,70.]),np.zeros(21,dtype=float)])
    A = (np.diag(np.ones(21))+np.diag(np.ones(20),1)+np.diag(np.ones(19),2))[3*np.arange(7)].astype(float)
    b = data[:,1][3*np.arange(7)]
    #solve the matrices
    sol=solvers.lp(matrix(c),G = matrix(G), h = matrix(h), A=matrix(A), b=matrix(b))
    return np.ravel(sol['x']), sol['primal objective']*(-1000.)







#if __name__ == "__main__":
    #print(prob1())
    #A = np.array([[1,2,1,1],[0,3,-2,-1]])
    #b = np.array([7,4])
    #print(l1Min(A, b))
    #print(prob3())
    #print(prob4())
    #print(l2Min(A, b))
    #print(prob6())
