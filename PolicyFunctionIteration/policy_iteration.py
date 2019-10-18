# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name>
<Class>
<Date>
"""
import numpy as np
import numpy.linalg as la
import gym
from gym import wrappers
# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3
P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
    """
    #initilize the true value for function
    v = np.zeros(nS)
    #Iterate only maxiter time at most
    for i in range(maxiter):
        #update the v
        v0 =np.copy(v)
        for s in range(nS):
            #Check all the states
            v[s] = np.max([np.sum([k[0]*(k[2]+beta*v0[k[1]]) for k in P[s][a]]) for a in range(nA)])
        
        #stop iterating if the approximation stops changing enough 
        if la.norm(v-v0,np.inf) < tol:
            break

    return v, i+1
        
    
# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    #initilize the true value for function
    pi = np.zeros(nS)
    #Iterate only nS time at most
    for s in range(nS):
        pi[s] = np.argmax([sum([k[0]*(k[2]+beta*v[k[1]]) for k in P[s][a]]) for a in range(nA)])
    return pi
        
            
        
    
# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy.
    
    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
    
    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    #initilize the true value for function
    best = [P[s][policy[s]] for s in range(nS)]
    v = np.zeros(nS)
    for i in range(3000):
        v0 = np.copy(v)
        
        v = np.array([sum([k[0]*(k[2]+beta*v[k[1]]) for k in a]) for a in best])
        #stop iterating if the approximation stops changing enough 
        if la.norm(v-v0) < tol:
            break
    return v
    
    
# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
        policy (ndarray): which direction to moved in each square.
    """
    #initilize the true value for function
    pi0 = np.ones(nS)
    
    for i in range(maxiter):
        #policy evaluation
        v = compute_policy_v(P, nS, nA,pi0, beta, tol)
        #policy improvement
        pi = extract_policy(P, nS, nA, v, beta)
        #stop iterating if the approximation stops changing enough 
        if la.norm(pi-pi0) < tol:
            break
        #update the pi
        pi0 = np.copy(pi)
    return v, pi, i+1
    
# Problem 5
def frozenlake(basic_case=True, M=1000):
    """ Finds the optimal policy to solve the FrozenLake problem.
    
    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns. 
    M (int): The number of times to run the simulation.
    
    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The average expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value functiono for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The average expected value for following the policy iteration optimal policy.
    """
    #part one
    #call enviroment
    env_name = 'FrozenLake-v0' if basic_case else 'FrozenLake8x8-v0'
    
    env = gym.make(env_name).env
    
    #get the P, nS, nA
    nS = env.nS
    nA = env.nA
    P = env.P
    
    #get different policies and value functions
    vi_policy = extract_policy(P,nS,nA, value_iteration(P,nS,nA)[0])
    pi_value_func, pi_policy, i = policy_iteration(P, nS, nA)
    
    #part two
    #run through different simulations
    vi_total_reward = 0 
    pi_total_reward = 0 
    for i in range(M):
        vi_total_reward += run_simulation(env, vi_policy)
        pi_total_reward = run_simulation(env, pi_policy)
    vi_total_reward /= M 
    pi_total_reward /= M 
    
    return vi_policy,vi_total_reward, pi_value_func, pi_policy, pi_total_reward
        
    
    
# Problem 6
def run_simulation(env, policy, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.
    
    Parameters:
    env (gym environment): The gym environment. 
    policy (ndarray): The policy used to simulate.
    beta float): The discount factor.
    
    Returns:
    total reward (float): Value of the total reward recieved under policy.
    """
    #initilize the reward and iter
    total_reward, i =0,0
    
    #get obs
    obs = env.reset()
    done = False
    
    #keep on going untill done
    while not done:
        #get the new obs
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += beta**i * reward
    return total_reward

if __name__ == "__main__":
    v,iter = value_iteration(P, 4 ,4)
    print(v)
    policy = extract_policy(P, 4 ,4, v, beta = 1.0)
    print(policy)
    print(compute_policy_v(P, 4 ,4, policy))
    print(policy_iteration(P, 4 ,4))