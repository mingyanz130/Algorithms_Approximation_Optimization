"""HW_6.8 and 6.9
<Mingyan Zhao>
<Math 321>
<11/05/2018>
"""
import numpy as np
#6.8
def prob1():
    Eps = [0.1, 0.01, 0.001]
    #variance = p*(1-p)
    count1 = 0
    count2 = 0
    count3 = 0

    answers = np.random.binomial(1000, .5, size = 100)
    means = answers/1000
    var = .25
    bounds = [var/(1000*i**2) for i in Eps]

    for i in means:

        if abs(i-.5) >= Eps[0]:
            count1 += 1
        if abs(i-.5) >= Eps[1]:
            count2 += 1
        if abs(i-.5) >= Eps[2]:
            count3 += 1
    count1 /= 100
    count2 /= 100
    count3 /= 100
    return "the bounds for epsilon are " + str(bounds) + "and respective proportion is " + str(count1) + ", " + str(count2) +", " + str(count3)


def prob2():
    #beta(1,9)
    Eps = [0.1, 0.01, 0.001]
    #variance = p*(1-p)
    count1 = 0
    count2 = 0
    count3 = 0
    answers = []
    mean = 1/(1+9)
    var = (1*9)/((1+9)**2*(1+9+1))

    for i in range(100):
        sum = 0
        for j in range(1000):
            sum += np.random.beta(1,9)
        answers.append(sum/1000)
    means = answers

    bounds = [var/(1000*i**2) for i in Eps]

    for i in means:

        if abs(i-mean) >= Eps[0]:
            count1 += 1
        if abs(i-mean) >= Eps[1]:
            count2 += 1
        if abs(i-mean) >= Eps[2]:
            count3 += 1
    count1 /= 100
    count2 /= 100
    count3 /= 100
    return "the bounds for epsilon are " + str(bounds) + "and respective proportion is " + str(count1) + ", " + str(count2) +", " + str(count3)
