"""
<Mingyan Zhao>
<Math 320>
<10/09/2018>
"""
from matplotlib import pyplot as plt
import time
import numpy as np

#4.1
def fibonacci_naive(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return fibonacci_naive(n-1) + fibonacci_naive(n-2)

def fibonacci_memoized(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        f = [None]*(n+1)
        f[0] = 1
        f[1] = 1
        for i in range(2, n + 1):
            f[i] = f[i-1] + f[i-2]
            
    return f[n]    
            
def fibonnacci_bottom_up(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        first = 1
        second = 1
        for i in range(2, n + 1):
            f = first + second
            first = second
            second = f            
    return f

def plot_fibonnacci():
    domain = range(0,34)
    
    times1 = []
    times2 = []
    times3 = []
    
    #record the start time and end time for each function
    for n in domain:
        start1 = time.time()
        fibonacci_naive(n)
        times1.append(time.time() -start1)
        
        start2 = time.time()
        fibonacci_memoized(n)
        times2.append(time.time() -start2)
        
        start3 = time.time()
        fibonnacci_bottom_up(n)
        times3.append(time.time() -start3)
    
    plt.plot(domain, times1, 'g.-', linewidth = 2, markersize = 15, label = "Naive")
    plt.plot(domain, times2, 'b.-', linewidth = 3, markersize = 10, label = "memoized")
    plt.plot(domain, times3, 'c.-', linewidth = 1, markersize = 5, label = "bottom_up")
    plt.legend(loc = "upper left")
    plt.show()  
    plt.savefig('fibonnacci.png', bbox_inches='tight')


#4.2

def change_naive(v, C = [1,5,10,25,100]):
      Opt = [0 for i in range(0, v+1)]
      sets = {i:[] for i in range(v+1)}
      n = len(C)
      for i in range(1, v+1):
            smallest = float("inf")
            for j in range(0, n):
                 if (C[j] <= i):
                       smallest = min(smallest, Opt[i - C[j]]) 
                       if smallest == Opt[i - C[j]]:
                            sets[i] = [C[j]] + sets[i-C[j]]
            Opt[i] = 1 + smallest 
            
      List = [0] * n
      for k in range(n):
          List[k] = sets[v].count(C[k])
          
      return Opt[v], List
def change_bottom_up(v, C = [1,5,10,25,100]):
    n = len(C)
    List = [0]*n
    C_ = sorted(C, reverse = True)    
    j = 0
    for i in C_:
        if v >= i:
            List[j] = int(v/i)
            v = v % i
        j += 1
    return sum(List), List[::-1]   
    
#4.3  
def change_greedy(v, C = [1,5,10,25,100]):
    n = len(C)
    List = [0]*n
    C_ = sorted(C, reverse = True)    
    j = 0
    for i in C_:
        if v >= i:
            List[j] = int(v/i)
            v = v % i
        j += 1
    return sum(List), List[::-1]   

def plot_change():
    domain = range(1,1000)
    
    times1 = []
    times2 = []
    times3 = []
    
    #record the start time and end time for each function
    for n in domain:
        start1 = time.time()
        change_naive(n)
        times1.append(time.time() -start1)
        
        start2 = time.time()
        change_bottom_up(n)
        times2.append(time.time() -start2)
        
        start3 = time.time()
        change_greedy(n)
        times3.append(time.time() -start3)
    
    plt.plot(domain, times1, 'g.-', linewidth = 1, markersize = 5, label = "Naive")
    plt.plot(domain, times2, 'b.-', linewidth = 1, markersize = 5, label = "Bottom_up")
    plt.plot(domain, times3, 'c.-', linewidth = 1, markersize = 5, label = "Greedy")
    plt.legend(loc = "upper left")
    plt.show()  
"""   
part two
The greedy solution is the optimal solution because the number coins with greater value will always
will always be less than the number of coins with smaller value. So it is always the optimal solution for 
the U.S coinage system.
"""