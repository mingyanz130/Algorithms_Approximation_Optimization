# -*- coding: utf-8 -*-
"""
<Mingyan Zhao>
<Math 320>
<09/20/2018>
"""


def euclidean_algorithm(x, y):
    #usedfor keeping records for division
    
    a, b, c, d = 1, 0, 0, 1
    
    #denomination cannot be zero
    while y != 0:
        #reassign the value for the calculation to continue
        q, x, y = x // y, y, x % y
        #record the x, y for the equation
        a, b = b, a - q * b
        c, d = d, c - q * d
    #return the gcd and x, y
    return  x, a, c