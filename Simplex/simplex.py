# simplex.py
"""Volume 2: Simplex.
<Name>
<Class>
<Date>

Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""
import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the tableau.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        #initilize the shape of A
        m,n = A.shape
        self.m, self.n = m,n
        #check if the system is feasible at the origin
        for i in b:
            if i < 0:
                raise ValueError("The problem is not feasible at the origin")
        #adding slack variables
        L = np.roll(np.arange(m+n),m)
        self.L = L
        #creating  a Tableau
        A_ = np.hstack((A, np.eye(m)))
        c_ = np.hstack((-c, np.zeros(m)))
        b_ = np.array([np.hstack((np.array([0]),b))])
        one = np.array([np.hstack((np.array([1]),np.zeros(m)))])
        T = np.vstack((c_.T,A_))
        T = np.hstack((b_.T,T,one.T))
        #store the tableau
        self.T =T


    def pivot(self):
        np.seterr(divide="ignore")
        #get the shape of the tableau
        m,n = self.T.shape
        first = self.T[0]


        for i in range(n):
            #find the first negtive value
            if first[i] < 0:
                #check the pivot column
                check1 = self.T[1:,0]
                check2 = self.T[1:,i]
                if np.all(check2)<0:
                    raise ValueError("The problem is not bounded")
                #check the colum with the smallest ratio
                mask = check2 > 0
                check3 = check1[mask]/ check2[mask]
                j = 0
                check4 = check1/check2
                mini = min(check3)
                while check4[j] != mini:
                    j += 1
                #return the indices
                return j+1,i

            # return the solution if all elements in the first row are  positive
            if i == n-1:
                return (self.T[0,0], dict(zip(self.L[:m],np.round(self.T[1:,0],decimals=3))),dict(zip(self.L[m-1:],np.zeros(n))))




    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        while True:
            #get the indices
            pivot = self.pivot()
            #check if the solution is already found
            if len(pivot) == 3:
                return pivot
            #get the indices of the variable
            i, j = pivot
            a ,b = list(self.L).index(i-1+self.n), list(self.L).index(j-1)
            #switch the index
            self.L[a], self.L[b]=self.L[b],self.L[a]

            #row operation on the tableau
            self.T[i] /= self.T[i,j]
            for k in range(len(self.T)):
                if k != i:
                    self.T[k] -= self.T[k,j]*self.T[i]



# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    #load the data
    data = np.load(filename)
    A, p, m, d = data["A"], data["p"],data["m"],data["d"]
    #form the data in the right format
    A_ = np.vstack((A,np.eye(4)))
    b = np.hstack((m,d))
    #using SimplexSolver
    solution = SimplexSolver(p,A_,b).solve()
    return np.array([solution[1][i] for i in range(4)])

"""
if __name__ == "__main__" :
    b = np.array([2,5,7])
    A = np.array([[1,-1],[3,1],[4,3]])
    c = np.array([3,2])

    print(SimplexSolver(c,A,b).solve())
    print(prob7())
"""
