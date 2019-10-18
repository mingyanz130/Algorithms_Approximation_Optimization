"""Volume 2:
<Mingyan Zhao>
<Math 322>
<01/09/2019>
"""

import numpy as np
import time

#problem 5

class Bandits:
    def __init__(self, probabilities, payouts):
        self.arms = len(payouts)
        self.prob = probabilities
        self.payouts = payouts
        self.success = [0] * self.arms
        self.failure = [0] * self.arms
        self.dic = dict()
    
    def pull(self, i):
        if np.random.uniform() <= self.prob[i]:
            self.success[i] += 1
            return (self.payouts[i],(1,0)) 
        else: 
            self.failure[i] += 1
            return (0,(0,1)) 
    def compute_R(self, M, r, beta):
        if r in self.dic.keys():
            return self.dic[r]
        else:
            R = np.zeros((M+1,M+1))
            for i in range(M+1):
                R[i,M-i] = 1/(1-beta)*max(i/(M),r)
                
            for k in range(1,M)[::-1]:
                for i in range(k+1):
                    R[i,k-i] = max((i/k)*(1+beta*R[i+1,k-i])+(k-i)/k*beta*R[i,k-i+1],r/(1-beta))
            self.dic[r] = R
            
            return R
        
    
    def gittens(self,J, states, M, K, beta=0.9):
        v = []
        for l, state in enumerate(states):
            i,j = state
            r = np.linspace(0,1,K+2)[1:-1]
            err = []
            for k in r:
                
                R = self.compute_R( M, k, beta)
                
                err.append(abs(k - (i*(1-beta)/(i+j)*(1+beta*R[i+1,j]) + j*(1-beta)/(i+j)*beta*R[i,j+1])))
            v.append(r[np.argmin(err)]*J[l])
        return v
                    
            


    
def prob_7(probabilities, payouts, K, T, M,beta):
    A = Bandits(probabilities, payouts)
    n = len(probabilities)
    states = [(1,1) for i in range(n)]
    pay = [0 for i in range(n)]
    for i in range(T):
        ind = np.argmax(A.gittens(payouts,states,M,K,beta))
        condition = A.pull(ind)
        states[ind] = (states[ind][0]+condition[1][0],states[ind][1]+condition[1][1])
        pay[ind] += condition[0]
        
    
    prob = [i[0]/(i[0]+i[1]) for i in states]
    print(states)
    return prob, pay
        


def pull(probabilities,payouts,i):
        if np.random.uniform() <= probabilities[i]:
            return (payouts[i],(1,0)) 
        else:        
            return (0,(0,1)) 
        
#problem 8
def thompson(states):
    prob = []
    for state in states:
        prob.append(np.random.beta(state[0],state[1]))
    
    return np.argmax(prob)

#problem 9
def prob_9(probabilities,payouts, T):
    n = len(probabilities)
    states = [(1,1) for i in range(n)]
    pay = [0 for i in range(n)]
    for i in range(T):
        ind = thompson(states)
        condition = pull(probabilities,payouts,ind)
        states[ind] = (states[ind][0]+condition[1][0],states[ind][1]+condition[1][1])
        pay[ind] += condition[0]
    prob = [i[0]/(i[0]+i[1]) for i in states]
    return prob, pay
    
#problem 10
def prob_10(probabilities, payouts, T):
    beta = 0.9
    K =99
    M = T+1
    times7 =[]
    times9 = []
    pay7 = []
    pay9 = []
    prob7 = np.zeros((T,len(probabilities)))
    prob9 = np.zeros((T,len(probabilities)))
    
    for j in range(20):
        start = time.time()
        vars1 = prob_7(probabilities, payouts, K, T, M,beta)
        times7.append(time.time()-start)
        prob7[j,:] = vars1[0]
        pay7.append(vars1[1])
        
        start = time.time()
        vars2 = prob_9(probabilities, payouts,T)
        times9.append(time.time()-start)
        prob9[j,:] = vars2[0]
        pay9.append(vars2[1])
    
    return (np.sum(times7)/20, np.sum(prob7,axis=0)/20, np.sum(pay7)/20),(np.sum(times9)/20, np.sum(prob9,axis=0)/20, np.sum(pay9)/20)
        
    
    

    


if __name__ == "__main__":
    A = Bandits([.5,.5,.5],[20,30,40])
    #print(A.pull(0))
    #print(A.compute_R(4,0.35,0.9))
    #print(A.gittens([1,1,1],[(6,2),(2,1),(1,1)],10,99))
    print(prob_7([.9,.7,.5],[20,30,40],99, 50, 51,0.9))
    #print(prob_9([.2,.5,.7],[1,1,1],20))
    #print(prob_10([.2,.5,.7],[1,1,1],100))
    
    
"""



17.10) Now that you have prob_7 and prob_9 to simulate both simulations, compare 
    the results. Create a function called prob_10(probabilities, payouts, T) (15-25 lines). 
    Assume beta=0.9, K=99, and M=T+1 when needed. This function should return 
    the mean run time, estimated probabilities, and payout averaged over 20 
    simulations for prob_7 and then prob_9. For example your output should have the form:

>>> res = prob_10(probabilities, payouts, T)
>>> res
((avg_time7, avg_est_probabilities7, avg_payout7), (avg_time9, avg_probabilities9, avg_payout9))

where avg_time and avg_payout are floats and avg_est_probabilities is an array with an entry for each arm.

"""