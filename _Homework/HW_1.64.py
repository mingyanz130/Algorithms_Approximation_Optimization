
"""
<Mingyan Zhao>
<Math 320>
<09/22/2018>
"""

#Problem 1.64 

"""
(i) The algorithm
a.Base case: If the the list has no element, return none, 
if the list has only one element, return the element.

b.If the list has more than one element, go to the middlepoint.
    If the next element if greater than the middle element, return the function 
    with the list of right half.
    If the next element is smaller than the middle element, return the function
    with the list of the left half.
c.  Continue the function until one element is left, which is the maximum.


"""
#(ii)
def prob(list):
    #start and the end of the list
    left = 0
    right = len(list) -1
    
    #the center of the list
    midpoint = (left + right)//2
    
    #base case
    if len(list) == 0:
        return None
    elif len(list) == 1:
        return list[0]    
    
    if list[midpoint] < list[midpoint+1]:
        return prob(list[midpoint+1:])
    elif list[midpoint] > list[midpoint+1]:
        return prob(list[:midpoint+1])
  
    
"""
(iii) According to the algorithm and  the function, a = 1, b = 2, d = 0
    then  b^d = a, then T(n) belongs to O(n^0*log(n))= O(log(n))
"""
    