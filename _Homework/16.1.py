#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mingyanzhao
"""
import numpy as np

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
            

def prob4(T, N, B, u=np.sqrt):
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

if __name__ == "__main__":
    print(prob4(3,4,0.9))