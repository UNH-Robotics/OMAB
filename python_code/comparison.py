import numpy as np
import matplotlib.pyplot as plt
#import seaborn
from math import sqrt, log
from random import random

# Make sure we have no type 3 fonts
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


## Helper methods

def bernoulli(p):
    """ Sample from Bernoulli distribution """
    if random() <= p: return 1
    else: return 0

## Evaluation method


def evaluate(method, horizon, runs):
    """
    Evaluates the multi-armed bandit method
    
    Parameters
    ----------
    method : class or constructor
        A class with choose and update methods
    horizon : int
        Horizon length
    runs : int, or list of tuples
        Configurations of bandits to run. 
        If it is an integer, then bandits are generated randomply according to 
        the uniform beta distribution.
        If it is a list of tuples, then each item is treated as a configuration
        for the two arms.
        
    Returns
    -------
    out : ndarray, matrix
        Each row is a single run and the entries are the cumulative regrets
        up to that point. Note that even this is a single run, the rewards used in 
        computing the regret are the expected values and not the actual realizations
    """
    
    if type(runs) == int:
        runs = (None,)*runs
    
    regrets = - np.ones( (len(runs), horizon) )

    for irun, run in enumerate(runs):        
        # generate problem 
        if run is None:
            pA = np.random.beta(1,1); pB = np.random.beta(1,1); 
        else:
            pA,pB = run
        maxp = max(pA, pB)
        # initialize
        losses = -np.ones(horizon); 
        m = method()
        # simulate
        for t in range(horizon):
            arm = m.choose(t+1)
            if arm == 0: p = bernoulli(pA)
            else: p = bernoulli(pB)            
            # update the algorithm
            m.update(arm,p)            
            # update the regret (using the expected regret)
            losses[t] = (maxp - (pA if arm==0 else pB))
        regrets[irun,:] = np.cumsum(losses)     
    return regrets
    
## Simple Methods
    
class UCB:
    """
    Upper confidence bound. Note the randomization on ties. This is
    just to make the method independent of the order of arms and the sensitivity
    with respect to the initialization.
    """
    def __init__(self,alpha = 2.0):
        self.Amean = 0; self.Bmean = 0; 
        self.Acount = 1; self.Bcount = 1;
        self.alpha = alpha
        
    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        valA = self.Amean + sqrt((self.alpha * log(t)) / (2*self.Acount) )
        valB = self.Bmean + sqrt((self.alpha * log(t)) / (2*self.Bcount) )
        # the UCB choice
        if valA > valB: return 0
        elif valA < valB: return 1
        else: return bernoulli(0.5)
    
    def update(self,arm, outcome):
        """ Updates the estimate for the arm outcome """
        if arm == 0:
            self.Amean = (self.Amean * self.Acount + outcome) / (self.Acount + 1)
            self.Acount += 1
        elif arm == 1:
            self.Bmean = (self.Bmean * self.Bcount + outcome) / (self.Bcount + 1)
            self.Bcount += 1
        else:
            raise RuntimeError("Invalid arm number")


class Thompson:
    """
    Thompson sampling
    """
    def __init__(self):
        # initialize prior values
        self.Acountpos = 1; self.Acountneg = 1;
        self.Bcountpos = 1; self.Bcountneg = 1;
    
    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        pA = np.random.beta(self.Acountpos, self.Acountneg)
        pB = np.random.beta(self.Bcountpos, self.Bcountneg)
        if pA > pB: return 0
        else: return 1
        
        
    def update(self,arm, outcome):
        """ Updates the estimate for the arm outcome """
        if arm == 0:
            if outcome == 1: self.Acountpos += 1
            else: self.Acountneg += 1
        elif arm == 1:
            if outcome == 1: self.Bcountpos += 1
            else: self.Bcountneg += 1
        else:
            raise RuntimeError("Invalid arm number")

## Gittins index


# loads the index as a global variable (to avoid reinit in every run)
gittins = {}
import pandas as pa
gittins_csv = pa.read_csv('gittins.csv')
for (p,n,v) in zip(gittins_csv.Positive, gittins_csv.Negative, gittins_csv.Index):
    gittins[(p,n)] = v

class Gittins:
    """
    Use Gittins index. Note the randomization when the values are tied. This is
    just to make the method independent of the order of arms and the sensitivity
    with respect to the initialization.
    """
    def __init__(self):
        # initialize prior values
        self.Acountpos = 1; self.Acountneg = 1;
        self.Bcountpos = 1; self.Bcountneg = 1;
    
    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        pA = gittins[(self.Acountpos, self.Acountneg)]
        pB = gittins[(self.Bcountpos, self.Bcountneg)]
        if pA > pB: return 0
        elif pA < pB: return 1
        else: return bernoulli(0.5)
        
    def update(self,arm, outcome):
        """ Updates the estimate for the arm outcome """
        if arm == 0:
            if outcome == 1: self.Acountpos += 1
            else: self.Acountneg += 1
        elif arm == 1:
            if outcome == 1: self.Bcountpos += 1
            else: self.Bcountneg += 1
        else:
            raise RuntimeError("Invalid arm number")
    
## Compute the mean regret

horizon = 100
trials = 5000

ucb_regrets = evaluate(UCB, horizon, trials)
thompson_regrets = evaluate(Thompson, horizon, trials)
gittins_regrets = evaluate(Gittins, horizon, trials)
    
## Plot the mean regret

plt.plot(ucb_regrets.mean(0), label='UCB')
plt.plot(thompson_regrets.mean(0), label='Thompson')
plt.plot(gittins_regrets.mean(0), label='Gittins')
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('Regret')
plt.grid()
plt.savefig('regrets.pdf')
plt.show()

## Compute regret as a function of delta (difference between the two arms)

ticks = 30
repetitions = 50

# init p values
runs = tuple((p1, p2) for p1 in np.linspace(0,1,ticks) for p2 in np.linspace(0,1,ticks) for r in range(repetitions) if p1 != p2) 
deltas = np.array(tuple(abs(pA-pB) for pA, pB in runs))

ucb_regrets = evaluate(UCB, horizon, runs)
thompson_regrets = evaluate(Thompson, horizon, runs)
gittins_regrets = evaluate(Gittins, horizon, runs)

## Plot dependence on delta

# average the 
shrunkdelta = deltas.reshape(-1, repetitions)[:,0]
shrunkprobs = np.array([max(p1,p2) for p1,p2 in runs]).reshape(-1, repetitions)[:,0]

def plot_curve(data, pos, name):
    plt.subplot(1,3,pos)
    plt.scatter(shrunkdelta,data[:,-1].reshape(-1,repetitions).mean(1)/(shrunkprobs*horizon)*100, s=10, c=shrunkprobs, edgecolors='face',cmap=matplotlib.cm.plasma)
    plt.ylim(-1,30)
    plt.xlabel('$\Delta$'); plt.ylabel('Mean propotional regret (\%)'); plt.title(name)
    plt.grid()

plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plot_curve(ucb_regrets, 1, "UCB")
plot_curve(thompson_regrets,2, "Thompson")
plot_curve(gittins_regrets, 3, "Gittins")

plt.savefig('proportional_regret.pdf')
plt.show()