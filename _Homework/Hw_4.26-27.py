# -*- coding: utf-8 -*-
"""Math 320
<Mingyan Zhao>
<Math 345>
<10/16/2018>
"""


#Problem 4.26
def knapsack(W, Iterms):
    
    set1 = []
    sets2 = Iterms.copy()
    v = 0
    w = W
    if w < min(Iterms[:][0]):
        return v, set1
    else:
        for i in Iterms:
            if w >= i[0]:
                v += i[1]
                w -= i[0]
                set1.append(i)
    return v, sets
    

                

