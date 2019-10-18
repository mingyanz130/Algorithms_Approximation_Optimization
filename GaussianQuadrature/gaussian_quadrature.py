# solutions.py
"""Volume 2: Gaussian Quadrature.
<Mingyan Zhao>
<Math 323>
<01/17/2019>
"""
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from scipy.sparse import diags
from numpy.linalg import eig
from scipy.stats import norm
from scipy.integrate import quad

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        #store thevalue base on the polytype
        self.n = n
        if polytype == 'legendre':
            self.wi = lambda x: 1
            self.type = "legendre"
        elif polytype == 'chebyshev':
            self.wi = lambda x: (1-x**2)**(1/2)
            self.type = "chebyshev"
        else:

            raise ValueError("polytype is not 'legendre' or 'chebyshev'.")
        #store the points and weights
        self.xint, self.w = self.points_weights(n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """

        if self.type == 'legendre':
            #initialize the Jacobi matrix
            beta = np.array([k**2/(4*k**2-1) for k in range(1,n)])
            beta = np.sqrt(beta)
            alpha = np.array([0 for i in range(n)])
            diagonal = np.array([beta,alpha,beta])
            J = diags(diagonal, [-1,0,1]).toarray()

            #find the eigenvalues and eigenvectors
            eval, evec = eig(J)
            xint = eval
            w = (evec[0,:])**2*2
        elif self.type == 'chebyshev':
            #initialize the Jacobi matrix
            beta = np.array([1/4 for i in range(n-1)])
            beta[0] *= 2
            beta = np.sqrt(beta)
            alpha = np.array([0 for i in range(n)])
            diagonal = np.array([beta,alpha,beta])
            J = diags(diagonal, [-1,0,1]).toarray()
            #find the eigenvalues and eigenvectors
            eval, evec = eig(J)
            w = (evec[0,:])**2*np.pi
            xint = eval
        return xint, w


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        g = lambda x: f(x)*self.wi(x)
        #calucute the sum by taking the inner product
        return self.w.T@g(self.xint)
    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        #define h
        h = lambda x: f((b-a)/2*x+ (a+b)/2)
        #return the intergral from eariler problem
        return (b-a)/2*self.basic(h)
    # Problem 6
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        #define h and g
        h = lambda x, y: f((b1-a1)*x/2+(a1+b1)/2,(b2-a2)*y/2+ (a2+b2)/2)
        g = lambda x, y: h(x,y)/(self.wi(x)*self.wi(y))
        #take the sum
        sum = 0
        for i in range(self.n):
            for j in range(self.n):
                sum += self.w[i]*self.w[j]*g(self.xint[i],self.xint[j])
        #scale the sum and return
        return (b1-a1)*(b2-a2)/4*sum

# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    f = lambda x: (1/np.sqrt(2*np.pi))*np.e**((-x**2)/2)
    #the correct answer
    standard = norm.cdf(2)-norm.cdf(-3)
    err1 = []
    err2 = []
    err3 = []
    #for different number of points, calculate the error
    domain = 5*np.arange(1,11)
    for n in domain:
        l = GaussianQuadrature(n,'legendre')
        err1.append(l.integrate(f,-3,2)-standard)

        c = GaussianQuadrature(n,'chebyshev')
        err2.append(c.integrate(f,-3,2)-standard)

        err3.append(quad(f,-3,2)[0]-standard)

    #plot the error against the number of points
    plt.semilogy(domain, np.abs(err1), label= "legendre")
    plt.semilogy(domain, np.abs(err2), label= "chebyshev")
    plt.semilogy(domain, np.abs(err3), label= "Scipy")
    plt.xlabel("number of points and weights")
    plt.ylabel("errors")
    plt.title("errors against number of points")

    plt.legend()
    plt.show()
