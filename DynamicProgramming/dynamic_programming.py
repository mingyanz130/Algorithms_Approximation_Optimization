# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Name>
<Class>
<Date>
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    #initialize the list
    chance = np.zeros(N)
    t = N - 1
    #update the list backward
    while t > 0:
        chance[t-1] = (t)/(t+1)*chance[t]+1/N
        t -= 1
    chance[0] = 1/N
    #return the maximum and index
    return np.max(chance), np.argmax(chance)

# Problem 2
def graph_stopping_times(N):
    """Graph the optimal percentage of candidates to date optimal
    and expected value for the marriage problem for n=3,4,...,M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for N.
    """
    #make two lists to store the data
    perc = []
    maxp = []
    #set up the range
    domain = np.arange(3,N+1)
    for i in domain:
        x,y = calc_stopping(int(i))
        perc.append(y/i)
        maxp.append(x)
    #graph
    plt.plot(domain,perc, label="the percentage of candidates")
    plt.plot(domain,maxp, label="the maximum probability")
    plt.legend()
    plt.show()
    return perc[-1]
    
    
        


# Problem 3
def get_consumption(N, u=np.sqrt):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        ((N+1,N+1) ndarray): The consumption matrix.
    """
    #partition vector whose entries correspond to possible amounts of cake
    W = [i/N for i in range(0,N+1)]
    
    #construct the cosumption matrix
    C = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(i+1):
            C[i,j] = u(W[i]-W[j])
    return C
            


# Problems 4-6
def eat_cake(T, N, B, u=np.sqrt):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    A = np.zeros((N+1,T+1))
    P = np.zeros((N+1,T+1))
    w = np.linspace(0,1,N+1)
    
    A[:,T] = u(w)
    P[:,T] = w
    
    for t in range(T)[::-1]:
        CV = np.zeros((N+1,N+1))
        for i in range(N+1):
            for j in range(N+1):
                wi = w[i]
                wj = w[j]
                if wi < wj:
                    CV[i,j] = 0
                else:
                    CV[i,j]=u(w[i]-w[j])+B*A[j,t+1]
        for i in range(N+1):
            w1 = w[i]
            max_ = max(CV[i,:])
            w2 = w[np.where(CV[i,:]==max_)[0][0]]
            P[i,t] = w1-w2
            A[i,t] = max(CV[i,:])
            
    return A,P
    


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((N,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    #initilize the matrices
    A, P = eat_cake(T, N, B, u=u)
    print(A,P)
    
    #initilize the starting points
    i = N
    #optimal path
    C = [P[i,0]]
    
    for t in range(1,T+1):
        #update the path
        cake_left = int(round(C[-1]*N))
        i -= cake_left
        C.append(P[i,t])
    return C
        
        
        
    
    
    
    
    
if __name__ == "__main__":
    print(graph_stopping_times(1000))
    #print(get_consumption(4))
    print(find_policy(3, 10, 0.75, u=np.sqrt))
