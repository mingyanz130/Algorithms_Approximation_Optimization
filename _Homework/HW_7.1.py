"""Volume 2: hw_7.1-7,5
<Mingyan Zhao>
<Math 321>
<11/17/2018>
"""
from scipy.stats import chi2
import matplotlib.pyplot as plt
import numpy as np
import random




def prob1():
    """
    using monte carlo methods
    """
    #Part A using monte carlo methods
    A = np.array([np.random.randn(10**i)**2 for i in [2,4,6]])
    bin = np.linspace(0,1,50)
    plt.subplot(131)
    plt.hist(A[0], bins=bin, density = True)
    plt.subplot(132)
    plt.hist(A[1], bins=bin, density = True)
    plt.subplot(133)
    plt.hist(A[2], bins=bin, density = True)

    #compute cdf
    answers_A1 = []
    answers_A2 = []
    answers_A3 = []
    for j in range(3):
        for k in [.5,1,1.5]:
            legth = len(A[j])
            answers_A1.append((np.sum(A[j]<=k)/legth,k,(j+1)*2))
            mean = np.average(A[j])

            L = A[j]-mean
            L = L**2
        answers_A2.append(mean)
        answers_A3.append(np.sum(L)/len(L))

    #part B built-in functions
    answers_B1 = []
    answers_B2 = []
    answers_B3 = []
    plt.subplot(131)
    plt.plot(bin, chi2.pdf(bin,1))
    plt.subplot(132)
    plt.plot(bin, chi2.pdf(bin,1))
    plt.subplot(133)
    plt.plot(bin, chi2.pdf(bin,1))
    answers_B1 = [chi2.cdf(i, df=1) for i in [0.5,1,1.5]]
    plt.show()
    answers_B2.append(chi2.mean(1))
    answers_B3.append(chi2.var(1))
    return "for k = 2, 4, 6, x= 0.5, 1.0, 1.5, the expected value is the first term, x is the second term, k is the third term" + str(answers_A1) + " These are the standard erros for each expected value " + str(answers_A2)+str(answers_A3)+str(answers_B1)+ " mean for chi_squared "+str(answers_B2)+str(answers_B3)


def prob2():
    list1 = []
    SE_list = []
    for i in [2,4,6]:
        # Number of darts that land inside.
        count = 0
        XY = []
        # Iterate for the number of darts.
        for k in range(10**i):
            X = np.random.uniform(low=-1)
            Y = np.random.uniform(low=-1)
            xy = X**2+Y**2
            XY.append(xy)
            if xy <= 1:
                count += 1
        pi = 4*count/(10**i)
        list1.append(pi)
        XY = np.array(XY)
        mean = np.average(XY)
        SE = np.sqrt(np.sum((XY-mean)**2)/(len(XY)-1))/np.sqrt(len(XY))
        SE_list.append(SE)

    return "the estimation of pi and their SE: " + str(list1) +"SE: "+str(SE_list)

def prob3(n = 10**5):
    #part 1

    x1 = np.random.uniform(0,2, n)
    hx = np.exp(np.cos(x1**2))
    mean1 = 2*np.mean(hx)
    err1 = np.sqrt((np.sum((hx-np.mean(hx))**2)/(n-1))/n)

    #part 2
    b = np.e
    count = 0
    x2 = np.random.uniform(0,2, n)
    y2 = np.random.uniform(0,b, n)
    list = []
    for i in range(n):
        if y2[i] <= np.exp(np.cos(x2[i]**2)):
            count += 1
            list.append(x2[i])
    mean2 = np.mean(list)
    err2 = np.sqrt((np.sum((x2-mean2)**2)/(n-1))/n)

    inter = 2*np.e*count/n
    return "the first result and SD: " + str(mean1)+ " "+str(err1), "the second result and SE: " + str(inter)+ " " + str(err2)

def prob4():
    #find the n untill 2 standard erros is less than 0.001
    i = 4
    n = 111500
    while True:
        b1 = np.random.beta(2,5,n)
        b2 = np.random.beta(20,55,n)
        dif = b1-b2
        percent = np.sum(dif<0)/ n
        s2x = np.sum((dif-np.mean(dif))**2)/(n-1)
        SE = np.sqrt(s2x/n)
        if 2*SE < 0.001:
            return("the number of sample: " + str(n))
            break
        n += 1

def prob5(n=10**5):
    #find the expected winning
    def trial(n_rolls=10):
        total_winning = 0
        rolls = np.array([random.randint(1,5) for i in range(10)])
        for roll in rolls:
            if roll ==1 or roll ==2:
                total_winning +=1
            elif roll == 3:
                total_winning += 2
            elif roll ==4:
                total_winning -=1
        return total_winning
    trials = np.array([trial() for x in range(n)])
    count = np.sum(trials < 0)
    mean = np.mean(trials)
    percent = count/n
    err = np.sqrt((np.sum((trials-mean)**2)/(n-1))/n)
    return "the rusult: " + str(percent) + " the SE: " + str(err)
