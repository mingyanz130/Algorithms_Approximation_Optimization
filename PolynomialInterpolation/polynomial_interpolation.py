# solutions.py
"""Volume 2: Polynomial Interpolation.
<Mingyan Zhao>
<Math 323>
<01/17/2019>
"""
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
from numpy import linalg as la

# Problem 1
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    x = sy.symbols("x")
    n = len(xint)
    den_ = np.ones((n,1))
    L = []
    #calculate the denominatoe and numerator
    for j in range(n):
        num = x - xint
        num = np.delete(num,j)
        num_ = 1
        for k in num:
            num_ *= k
        den = (xint[j]-xint)
        den = np.delete(den, j)
        den_[j] = np.prod(den)
        L.append(num_/den_[j])
    p = 0
    #take the sum of the function
    for i in range(n):
        p += yint[i]*L[i]
    #turn it into a lambda function
    p = sy.lambdify(x, p, "numpy")
    return p(points)


# Problems 2 and 3
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        n = len(xint)
        self.n = n
        self.xint = xint
        self.yint = yint                  # Number of interpolating points.
        w = np.ones(n)                  # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        self.C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / self.C
            temp = temp[shuffle]        # Randomize order of product.
            w[j] /= np.product(temp)
        self.w = w


    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        #form the denominator and numerator
        v = lambda x: np.prod(x-self.xint)
        num = lambda x: np.sum(self.w*self.yint/(x-self.xint))
        den = lambda x: np.sum(self.w/(x-self.xint))
        #sum up all the pointss
        p = np.array([num(x)/den(x) for x in points])
        #calculate the domain
        for i in range(len(points)):
            for j in range(self.n):
                if points[i] == self.xint[j]:
                    p[i] = self.yint[j]
        return p

    # Problem 3
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        #extend the points list
        self.xint = np.append(self.xint, xint)
        self.yint = np.append(self.yint, yint)
        #renew the weights list
        n = len(self.xint)
        m = len(self.w)
        w = np.ones(n)
        w[:m] = self.w
        #
        C = (np.max(self.xint) - np.min(self.xint)) / 4
        shuffle = np.random.permutation(n-1)

        for j in range(m, n):
            temp = (self.xint[j] - np.delete(self.xint, j)) / self.C
            temp = temp[shuffle]        # Randomize order of product.
            w[j] /= np.product(temp)
            #update the existing weights
        for j in range(m):
            temp = (self.xint[j] - xint) / self.C
            w[j] /= np.product(temp)

        self.w = w

#used for testing Barycentric class
def test():
    f = lambda x: 1/(1+25*x**2)
    domain = np.linspace(-1,1,1000)
    xint = np.linspace(-1,1,8)
    poly = Barycentric(xint, f(xint))
    plt.subplot(221)
    plt.plot(domain, lagrange(xint, f(xint), domain))
    plt.plot(domain, f(domain))

    plt.subplot(222)

    plt.plot(domain, poly(domain))
    plt.plot(domain, f(domain))

    plt.subplot(223)
    add_ = np.linspace(-0.99,0.97,3)
    poly.add_weights(add_,f(add_))
    plt.plot(domain, poly(domain))
    plt.plot(domain, f(domain))


    plt.show()



# Problem 4
def prob4():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #initialize the domain and function
    domain = np.linspace(-1,1,400)
    f = lambda x: 1/(1+25*x**2)
    err1 = []
    err2 = []
    #test for different n value
    for n in 2**np.arange(2,9):
        #test the err between build_in function and original function
        poly = BarycentricInterpolator(np.linspace(-1,1,n),f(np.linspace(-1,1,n)))
        err1.append(la.norm(f(domain)-poly(domain), ord=np.inf))
        #test the err between build_in function with chebshev points and original function
        points = np.array([np.cos(i*np.pi/n) for i in range(n+1)])
        cheb = BarycentricInterpolator(points,f(points))
        err2.append(la.norm(f(domain)-cheb(domain), ord=np.inf))
    plt.loglog(2**np.arange(2,9), err1, "-o", basex = 2, label = "equally spaced points")
    plt.loglog(2**np.arange(2,9), err2, "-o", basex = 2, label = "Chebyshev extremal points")
    plt.legend()
    plt.title("Errors")
    plt.show()

# Problem 5
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    #Obtain the Chebyshev coefficients
    x=np.real(np.fft.fft(f(np.array([np.cos(i*np.pi/n) for i in range(2*n)]))))[:n]/(2*n)
    x[1:n] = 2*x[1:n]
    return x



# Problem 6
def prob6(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    #load the data
    data = np.load("airdata.npy")
    #initialize the function and domain
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    #plot the origin data and interpolated data
    poly = BarycentricInterpolator(domain[temp2], data[temp2])
    plt.subplot(211)
    plt.plot(data)
    plt.title("Original")
    plt.subplot(212)
    plt.plot(poly(domain))
    plt.title("Interpolation")

    plt.show()
