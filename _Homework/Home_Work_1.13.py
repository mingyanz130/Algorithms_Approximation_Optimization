"""
Name: Mingyan Zhao
Class: Math 320
Date: 09/08/2018
"""

"""
1.13
(1)
"""
def find_min(L):
    """
    find the index of the minimum in the list L of integers.
    """
    
    # Set initial values
    min_val = L[0] #minimum
    n = len(L) #list length
    i = 1 #counter to increase through L.
    p = 0 # index number
    while i < n:
        if L[i] < min_val:
            min_val = L[i]
            p = i #update the index of the minimum
        i += 1 #Increment i
    
    return p

"""
(2) The spatial complexity is n since only one list with length n is 
used and stored.
The temporal compexity: 
    From line 12 to 15, there are 6 primitive operations.
    In line 16, it is a while loop, it has n operations.(n-1 success and 1 fail)
    In line 17, worst case: it goes through the if loop, it has 7(n-1) 
    primitive operations.
    Best case: it does not go through if, it has 4(n-1) primitive operations.
    
    In total, the temporal complexity is 8n-1(worst case) or 5n + 2(best case).

"""

#1.14
"""
(1)
It only needs (n-1) times because if the last number is the biggest, it dose 
not need to be swapped. If it is not, it will be swaped.Eventually, the last 
will still be the greatest.
"""
def selection_sort(L): 
    #set initial values    
    i = 0
    n = len(L)  
    #copy list L
    P = list.copy(L)
    p = find_min(L)
    
    while i < n - 1:
        #swith the smallest number with the ith number
        m = L[p] # smallest number
        L[p] = L[i]
        L[i] = m
        #find where the smallest number in P
        p = find_min(P)
        #delete the smallest number in P
        P.pop(p)    
        #find the next smallest number after deleting the smallest number
        p = find_min(P) # index of the minimum        
        m = P[p] #minimum       
        p = L.index(m)
        i += 1
    return L

"""
(3) 
The spatial complexity would be 2n since two lists are stored.
The temporal complexity:
    From line 52 - line 55, there are 5 primitive operations.
    Line 56 called a function and assigned a value. It has 8(n-1)+1,which 
    is 8n-7.
    Line 58, (n-1)success and 1 fail.
    line 60 to line62, 7n operations.
    line 64-68, it called function twice, it has (16n-15)n operations.
    line 69-71, there are 6n operations.
    
    In total, there are 16n^2+7n-2 primitive operations.
    
"""