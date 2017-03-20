#!/bin/python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
from math import sqrt, log
import tqdm
import random


matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


## Helper methods

def bernoulli(p):
    """ Sample from Bernoulli distribution """
    if random.random() <= p:
        return 1
    else:
        return 0
        

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
        runs = (None,) * runs

    regrets = - np.ones((len(runs), horizon))

    for irun, run in enumerate(tqdm.tqdm(runs)):
        # generate problem 
        if run is None:
            pA = np.random.beta(1, 1);
            pB = np.random.beta(1, 1);
        else:
            pA, pB = run
        maxp = max(pA, pB)
        # initialize
        losses = -np.ones(horizon);
        m = method()
        # simulate
        for t in range(horizon):
            arm = m.choose(t + 1)
            if arm == 0:
                p = bernoulli(pA)
            else:
                p = bernoulli(pB)
            # update the algorithm
            m.update(arm, p)
            # update the regret (using the expected regret)
            losses[t] = (maxp - (pA if arm == 0 else pB))
        regrets[irun, :] = np.cumsum(losses)
    return regrets        
    
    
## Standard methods            

class UCB:
    """
    Upper confidence bound. Note the randomization on ties. This is
    just to make the method independent of the order of arms and the sensitivity
    with respect to the initialization.
    """

    def __init__(self, alpha=2.0):
        self.Amean = 0.5
        self.Bmean = 0.5
        self.Acount = 1
        self.Bcount = 1
        self.alpha = alpha

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        
        valA = self.Amean + sqrt((self.alpha * log(t)) / (2 * self.Acount))
        valB = self.Bmean + sqrt((self.alpha * log(t)) / (2 * self.Bcount))
        
        #if(t < 10):
        #    print('UCB', valA, valB)
        
        if valA > valB:
            return 0
        elif valA < valB:
            return 1
        else:
            return bernoulli(0.5)

    def update(self, arm, outcome):
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
        self.Acountpos = 1
        self.Acountneg = 1
        self.Bcountpos = 1
        self.Bcountneg = 1

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        pA = np.random.beta(self.Acountpos, self.Acountneg)
        pB = np.random.beta(self.Bcountpos, self.Bcountneg)
        if pA > pB:
            return 0
        else:
            return 1

    def update(self, arm, outcome):
        """ Updates the estimate for the arm outcome """
        if arm == 0:
            if outcome == 1:
                self.Acountpos += 1
            else:
                self.Acountneg += 1
        elif arm == 1:
            if outcome == 1:
                self.Bcountpos += 1
            else:
                self.Bcountneg += 1
        else:
            raise RuntimeError("Invalid arm number")    
            
## Gittins index

# loads the index as a global variable (to avoid reinit in every run)
gittins = {}
import pandas as pa

gittins_csv = pa.read_csv('valuecomputation/gittins.csv')
for (p, n, v) in zip(gittins_csv.Positive, gittins_csv.Negative, gittins_csv.Index):
    gittins[(p, n)] = v


class Gittins:
    """
    Use Gittins index. Note the randomization when the values are tied. This is
    just to make the method independent of the order of arms and the sensitivity
    with respect to the initialization.
    """

    def __init__(self):
        # initialize prior values
        self.Acountpos = 1
        self.Acountneg = 1
        self.Bcountpos = 1
        self.Bcountneg = 1

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        pA = gittins[(self.Acountpos, self.Acountneg)]
        pB = gittins[(self.Bcountpos, self.Bcountneg)]
        if pA > pB:
            return 0
        elif pA < pB:
            return 1
        else:
            return bernoulli(0.5)

    def update(self, arm, outcome):
        """ Updates the estimate for the arm outcome """
        if arm == 0:
            if outcome == 1:
                self.Acountpos += 1
            else:
                self.Acountneg += 1
        elif arm == 1:
            if outcome == 1:
                self.Bcountpos += 1
            else:
                self.Bcountneg += 1
        else:
            raise RuntimeError("Invalid arm number")

## Plot confidence intervals

def plot_confidence(data, *args, **kwargs):
    """ 95% confidence interval"""
    x = np.arange(data.shape[1])
    mean = data.mean(0)
    sigma = data.std(0) / np.sqrt(data.shape[0])
    
    z = plt.plot(x,mean, *args, **kwargs)
    # make sure that the color is consistent
    kwargs['color'] = z[0].get_color()
    # remove label
    if 'label' in kwargs:
        del kwargs['label']
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([mean - 1.96 * sigma,
                        (mean + 1.96 * sigma)[::-1]]),alpha=0.3, *args, **kwargs)

#plot_confidence(ucb_regrets, '-', label='UCB')
#plt.legend()
#plt.show()
