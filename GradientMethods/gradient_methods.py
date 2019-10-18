# solutions.py
"""Volume 2: Gradient Descent Methods.
<Mingyan Zhao>
<Math 323>
<01/17/2019>
"""
import numpy as np
import numpy.linalg as la
from autograd import elementwise_grad
import scipy.optimize as opt
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=10000):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #covert input into floats
    x0 = np.array(x0, dtype=np.float)
    for i in range(maxiter):
        #find the minimal alpha
        find = lambda a: f(x0-a*Df(x0))
        a = opt.minimize_scalar(find)
        #calculate next term
        x1 = x0 - a.x*Df(x0).T
        #check the condition
        if la.norm(Df(x1), np.inf) < tol:
            return x1, True, i+1
        else:
            #update the number
            x0 = x1
    return x1, False, maxiter






# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #update the numbers
    #number of iterations
    n = len(x0)
    r0 = Q@x0-b

    d0 = -r0
    converged = False
    for i in range(n+1):
        #update the formula
        a0 = (r0.T@r0) / (d0.T@Q@d0)
        x1 = x0 + a0*d0
        r1 = r0 + a0*Q@d0
        b1 = (r1.T@r1)/(r0.T@r0)
        d1 = -r1+b1*d0

        #update the number
        r0 = r1
        d0 = d1
        x0 = x1

        #check if it converged
        if la.norm(r0) < tol:
            converged = True
            break

    return x1, converged, i


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-15, maxiter=10000):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #update the numbers
    #number of iterations

    r0 = -df(x0).T
    d0 = r0
    #initilize the function
    find = lambda a: f(x0+a*d0)
    a0 = opt.minimize_scalar(find).x
    x1 = x0 + a0*d0

    #
    for i in range(1,maxiter):
        #update the formula
        r1 = -df(x1).T
        b1 = (r1.T@r1)/(r0.T@r0)
        d1 = r1 + b1*d0

        #find the optimal step
        find = lambda a: f(x1+a*d1)
        a1 = opt.minimize_scalar(find).x
        x2 = x1 + a1*d1
        #update the number
        x1 = x2
        r0 = r1
        d0 = d1
        #check if it converged
        if la.norm(r1) < tol:
            return x2, True, i+1

    return x2, False, maxiter


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    #load the data
    data= np.loadtxt("linregression.txt")
    #save the shape and the matrix
    size = data.shape
    A = data.copy()

    #update A and b
    one = np.ones(size[0])
    b = data[:,0]
    A[:,0] = one
    #return the solution to the corresponding Normal Equations.
    return conjugate_gradient(A.T@A, A.T@b, x0)[0]






# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """

        #set up the function
        f = lambda b: np.sum(np.log(1+np.exp(-(b[0] + b[1]*x)))+(1-y)*(b[0]+b[1]*x))

        #store the beta values
        self.b = opt.fmin_cg(f,guess, disp = False)


    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        #return the prediction at x
        return 1/(1+np.exp(-(self.b[0]+self.b[1]*x)))


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #load the data
    data= np.load("challenger.npy")
    #get the temperature and indicator
    x = data[:,0]
    y = data[:,1]
    #initialize the guess
    guess = np.array([20,-1])
    a = LogisticRegression1D()
    #fit the model
    a.fit(x,y,guess)

    #set up the domain and prediction
    domain = np.linspace(30,100,100)
    #plot the graph
    plt.plot(domain, [a.predict(i) for i in domain], c = "orange", label = "")
    plt.scatter([31], [a.predict(31)], c = "green", label = "P(Damage) at Launch")
    plt.scatter(x,y, label = "Previous Damage")

    #titles and labels
    plt.title("Probability of O-Ring Damage")
    plt.xlabel("Temperature")
    plt.ylabel("O-Ring Damage")
    plt.legend()
    plt.show()

    return [a.predict(31)]
