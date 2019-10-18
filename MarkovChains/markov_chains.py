# markov_chains.py
"""Volume II: Markov Chains.
<Mingyan Zhao>
<Math 321>
<11/05/2018>
"""

import numpy as np
from scipy import linalg as la


# Problem 1
def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    #initial the matrix
    x = np.random.random((n,n))
    #normalize the matrix
    x = x/x.sum(axis=0)
    return x

# Problem 2
def forecast(days):
    """Forecast tomorrow's weather given that today is hot."""
    transition = np.array([[0.7, 0.6],[0.3, 0.4]])
    #initialize the state and list
    x = 0
    list = []
    #making loops for days given
    for i in range(days):
        #get the prediction based on current day, record the state
        x = np.random.binomial(1, transition[1, x])
        list.append(x)

    return list


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    #initialize the state and list
    weather = np.array([[0.5,0.3,0.1,0],[0.3,0.3,0.3,0.3],[0.2,0.3,0.4,0.5],[0,0.1,0.2,0.2]])
    list = []
    x = 1
    #making loops for days given
    for i in range(days):
        #get the prediction based on current day, record the state
        x = np.random.multinomial(1, weather[:,x])
        #update the state based on the index
        x = np.argmax(x)
        list.append(x)
    return list

# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    #initialize the state and list
    m, n = np.shape(A)
    x0 = np.random.random((n,1))
    x0 = x0/x0.sum()

    #take th elimitation
    for i in range(N):
        x1 = A@x0
        x2 = x0
        x0 = x1
        #check if it meets the qualification
        if la.norm(x1-x2) < tol:
            return x1

    raise ValueError("A^k dose not converge.")




# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        """
        self.names = set()

        with open(filename, "r") as word_list:
            words = word_list.read().split(' ')
        self.name = set(word)
        n = len(self.name)
        """
        #save each sentences
        with open(filename, 'r') as myfile:
            contents = myfile.readlines()[0:-1]

        self.contents = contents
        #initialize the states
        self.state = ["$start"]

        #update the states
        for i in range(len(contents)):
            lines = str(contents[i]).strip().split(" ")
            for j in range(len(lines)):
                if lines[j] not in self.state:

                    self.state.append(lines[j])

        self.state = self.state + ["$end"]

        #initialize the transition matrix
        n = len(self.state)
        A = np.zeros((n,n))

        #Add 1 to the entry of the transition matrix corresponding to
        #transitioning from the start state to the first word of the sentence.
        for i in range(len(contents)):
            lines = str(contents[i]).strip().split(" ")
            a = self.state.index(lines[0])
            A[a,0] += 1

        #Add 1 to the entry of the transition matrix corresponding to
        #transitioning from the last word of the sentence to the stop state.
        for i in range(len(contents)):
            lines = str(contents[i]).strip().split(" ")
            a = self.state.index(lines[-1])
            A[-1,a] += 1

        #Add 1 to the entry of the transition matrix corresponding to
        #transitioning from state x to state y.
        for i in range(len(contents)):
            lines = str(contents[i]).strip().split(" ")
            for k in range(0, len(lines)-1):
                a = self.state.index(lines[k])
                b = self.state.index(lines[k+1])
                A[b,a] += 1

        A[-1,-1] = 1
        #normalize the matrix
        A =  A/A.sum(axis=0)
        self.A = A



    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        list = []

        n = len(self.state)


        x = 0
        #loop the process untill it goes to the end
        while True:
            #get the prediction based on current state, record the state
            col = np.random.multinomial(1, self.A[:,x])

            #update the state based on the index
            x = np.argmax(col)

            #return the string when it reaches to end
            if self.state[x] == "$end":
                return " ".join(list)
            else:
                list.append(self.state[x])
