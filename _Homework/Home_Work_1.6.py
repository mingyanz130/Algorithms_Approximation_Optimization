
"""
Name: Mingyan Zhao
Class: Math 320
Date: 09/06/2018
"""
def subtract(a,b):
    """
    Subtract two numbers, where each number is input as a list of
    single-digit intergers, e.g., [1,2,3] = 123
    """
    """
    converting two inputs into number to compare which is greater, 
    we will always subtract a smaller number from the greater number.
    """
    j = len(a) - 1
    p = j
    i = 0
    A = 0
    while i <= p:
        A +=  a[i] * 10 **(j)
        i += 1
        j -= 1
        
    o = len(b) - 1
    q = o
    s = 0
    B = 0
    while s <= q:
        B +=  b[s] * 10 **(o)
        s += 1
        o -= 1  
    """
    """
    delta = abs(len(a)-len(b))
    
    if len(a) <= len(b):
        a = delta * [0] + a
    else:
        b = delta * [0] + b
            
    if A >= B:
        i = len(a) - 1  
        carry = 0
        while i >= 0:
            if a[i] < b[i]:
              a[i] = a[i] + 10 - b[i] + carry
              carry = -1  
            else:   
              a[i] = a[i] - b[i] + carry
              carry = 0            
            i -= 1  
        return a
    else:
        
        i = len(b) - 1   
        carry = 0
        while i >= 0:
            if b[i] < a[i]:
              b[i] = b[i] + 10 - a[i] + carry
              carry = -1  
            else:   
              b[i] = b[i] - a[i] + carry
              carry = 0    
            i -= 1 
        b[0] = -b[0]    
        return b
"""
(2)
Both temporal and spatial complexity of this algorithm are O(n); 
where n is the length of the longer list.The number of primitive 
operations is at most O(n), the memory required is bound by the 
number of digits it takes to represent n. each of these contributes 
at most log2(n < n,which is O(n).)
"""