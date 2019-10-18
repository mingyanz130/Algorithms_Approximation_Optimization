# -*- coding: utf-8 -*-
"""
<Mingyan Zhao>
<Math 320>
<09/15/2018>
"""
import numpy as np
import time

# 1.26
def prob26(k):
    """
    define random matrices A and B of size 2^k * 2^k and a 
    column vector X of length 2^k, each entry is a integer between 0 and 101
    """
    A = np.random.random_integers(0, 101, (2**k,2**k))
    B = np.random.random_integers(0, 101, (2**k,2**k))
    X = np.random.random_integers(0, 101, (2**k,1)) 
    
    #time (AB)X
    start_time1 = time.time()
    a = (A@B)@X
    end_time1 = time.time()
    #time A(BX)
    start_time2 = time.time()
    b = A@(B@X)
    
    end_time2 = time.time()
    # calculate the time elapsed
    time1 = end_time1 - start_time1
    time2 = end_time2 - start_time2
    #ratio = time of (AB)X / time of A(BX)
    return time1, time2, time1 / time2


"""
when k <= 10, the calculation time is zero expressed by the program. 
when k >= 11, as k increases, the ratio of the times of the two computations 
increases.

For (AB)X, the temporal complexity equals 2^k * 2^k *(2^k + 2^k - 1) + 2^k *(2^k + 2^k - 1) = 
2^(3k+1) + 2^(2k) - 2^k
for A(BX), the temporal complexity equals 2^k * (2^k + 2^k -1) + 2^k * (2^k + 2^k -1)
=2^(2k+2) - 2^(k+1) 
so the first one is greater than the second and it has higher order, the ratio should increase. 
""" 

#1.27
def prob27(k):
    """
    define random vectors U,V and X of length 2^k, each entry is a integer between 0 and 101
    """
    U = np.random.random_integers(0, 101, (2**k,1))
    V = np.random.random_integers(0, 101, (2**k,1))
    X = np.random.random_integers(0, 101, (2**k,1)) 
    I = np.eye(2**k)
    
    #time (I+U*V^T)*X
    start_time1 = time.time()
    a = (I + U@V.T)@X
    end_time1 = time.time()
    #time X + U*(V^T*X)
    start_time2 = time.time()
    b = X + U@(V.T@X)
    
    end_time2 = time.time()
    # calculate the time elapsed
    time1 = end_time1 - start_time1
    time2 = end_time2 - start_time2
    #ratio = time of (AB)X / time of A(BX)
    return time1, time2, time1 / time2

"""
To show the equation,
(I+U*V^T)*X = I*X + U*V^T*X = X + U*(V^T*X)
(ii)
when k <= 11, the calculation time is zero expressed by the program. 
when k >= 12, as k increases, the ratio of the times of the two computations 
increases.

For (I+U*V^T)*X, the temporal complexity equals 2^k * 2^k + 2^k * 2^k + 2^k * 2^k * (2^k +2^k-1) 
=2^(3k+1) + 2^(2k)
for X + U*(V^T*X), the temporal complexity equals 2^k + 2^k -1 + 2^k + 2^k = 2^(k+2)-1
=2^(2k+2) - 2^(k+1) 
so the first one is greater than the second and it has higher order, the ratio should increase.
"""





  