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


## New Methods


class OptimisticLookAhead:
    """
    Optimistic Look Ahead inspired on OGI paper by Gutin & Farias
    """

    def __init__(self):
        # initialize prior values
        self.Acountpos = 1;
        self.Acountneg = 1;
        self.Bcountpos = 1;
        self.Bcountneg = 1;
        self.betasamplecount = 100

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """

        tRemain = horizon - ((self.Acountpos + self.Acountneg + self.Bcountpos + self.Bcountneg) - 4)

        # TODO: WE NEED TO EXPLAIN THIS!!!
        discount = 0.9 if self.betasamplecount>=100 else np.log2(self.betasamplecount)/10
        tRemain = (1 - discount ** tRemain) / (1 - discount)

        v01 = np.mean(
            np.max(np.row_stack((np.random.beta(self.Acountpos + 1, self.Acountneg, size=self.betasamplecount),
                                 np.random.beta(self.Bcountpos, self.Bcountneg, size=self.betasamplecount))),
                   axis=0)) * tRemain

        v02 = np.mean(
            np.max(np.row_stack((np.random.beta(self.Acountpos, self.Acountneg + 1, size=self.betasamplecount),
                                 np.random.beta(self.Bcountpos, self.Bcountneg, size=self.betasamplecount))),
                   axis=0)) * tRemain

        v11 = np.mean(np.max(np.row_stack((np.random.beta(self.Acountpos, self.Acountneg, size=self.betasamplecount),
                                           np.random.beta(self.Bcountpos + 1, self.Bcountneg,
                                                          size=self.betasamplecount))), axis=0)) * tRemain

        v12 = np.mean(np.max(np.row_stack((np.random.beta(self.Acountpos, self.Acountneg, size=self.betasamplecount),
                                           np.random.beta(self.Bcountpos, self.Bcountneg + 1,
                                                          size=self.betasamplecount))), axis=0)) * tRemain

        pA = self.Acountpos / (self.Acountpos + self.Acountneg)
        valueA = pA * (1 + v01) + (1 - pA) * v02

        pB = self.Bcountpos / (self.Bcountpos + self.Bcountneg)
        valueB = pB * (1 + v11) + (1 - pB) * v12

        if valueA > valueB:
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

gittins_csv = pa.read_csv('gittins.csv')
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


## Simple methods            

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


## Lookead value function

valuefunction = {}
import pandas as pa

valuefunction_csv = pa.read_csv('valuecomputation/ucb_value.csv')
for (t, p, n, v) in zip(valuefunction_csv.Time, valuefunction_csv.Positive, valuefunction_csv.Negative, valuefunction_csv.Value):
    valuefunction[(t,p,n)] = v


class ValueFunction:
    """
    Use one-step lookahead with a *linearly separable* value function which
    is precomputed for eacha arm separately
    """

    def __init__(self):
        # initialize prior values
        self.Acountpos = 1;
        self.Acountneg = 1;
        self.Bcountpos = 1;
        self.Bcountneg = 1;

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """

        # the pre-computed value function is 0-based! (t=0 is the first time step)
        vApos = valuefunction[(t,self.Acountpos+1,self.Acountneg)] + \
                    valuefunction[(t,self.Bcountpos,self.Bcountneg)]
        vAneg = valuefunction[(t,self.Acountpos,self.Acountneg+1)]  + \
                    valuefunction[(t,self.Bcountpos,self.Bcountneg)]  
        vBpos = valuefunction[(t,self.Bcountpos+1,self.Bcountneg)] + \
                    valuefunction[(t,self.Acountpos,self.Acountneg)]
        vBneg = valuefunction[(t,self.Bcountpos,self.Bcountneg+1)] + \
                    valuefunction[(t,self.Acountpos,self.Acountneg)]

        pA = self.Acountpos / (self.Acountpos + self.Acountneg)
        qvalueA = pA * (1 + vApos) + (1 - pA) * vAneg
        
        pB = self.Bcountpos / (self.Bcountpos + self.Bcountneg)
        qvalueB = pB * (1 + vBpos) + (1 - pB) * vBneg

        # adjust value function by subtracting the value of no-action
        qvalueA -= valuefunction[(t,self.Acountpos,self.Acountneg)] + \
                    valuefunction[(t,self.Bcountpos,self.Bcountneg)]
        
        qvalueB -= valuefunction[(t,self.Acountpos,self.Acountneg)] + \
                    valuefunction[(t,self.Bcountpos,self.Bcountneg)]
        #if(t < 10):
        #    print('VFA', qvalueA, qvalueB)

        if qvalueA > qvalueB:
            return 0
        elif qvalueA < qvalueB:
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



## Compute the mean regret

horizon = 99
trials = 100

#np.random.seed(0)
#random.seed(0)
ucb_regrets = evaluate(UCB, horizon, trials)
#np.random.seed(0)
#random.seed(0)
vf_regrets = evaluate(ValueFunction, horizon, trials)

thompson_regrets = evaluate(Thompson, horizon, trials)
#ola_regrets = evaluate(OptimisticLookAhead, horizon, trials)
gittins_regrets = evaluate(Gittins, horizon, trials)


## Plot the mean regret
plt.figure(num=2, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(ucb_regrets.mean(0), label='UCB')
plt.plot(thompson_regrets.mean(0), label='Thompson')
#plt.plot(ola_regrets.mean(0), label='OptimisticLookahead')
plt.plot(vf_regrets.mean(0), '--', label='ValueFunction')
plt.plot(gittins_regrets.mean(0), label='Gittins')
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('Regret')
plt.grid()
#plt.savefig('regrets.pdf')
plt.show()

## Confidence interval
def calc_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    e = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, e

ymean = []
yconf = []
for t in range(horizon):
    m, e = calc_confidence_interval(ola_regrets[:, t])
    ymean.append(m)
    yconf.append(e)

plt.figure(num=1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(x=np.arange(0, 100, 1), y=ymean, yerr=yconf)
plt.legend(loc='upper left')
plt.xlabel('Horizon')
plt.ylabel('Regret with cofidence interval')
plt.title("Confidence interval on the mean regret of optimistic lookahead")
plt.grid()
plt.savefig('confidence_interval.pdf')
plt.show()

## Compute regret as a function of delta (difference between the two arms)

ticks = 30
repetitions = 500

# init p values
runs = tuple(
    (p1, p2) for p1 in np.linspace(0, 1, ticks) for p2 in np.linspace(0, 1, ticks) for r in range(repetitions) if
    p1 != p2)
deltas = np.array(tuple(abs(pA - pB) for pA, pB in runs))

ucb_regrets = evaluate(UCB, horizon, runs)
thompson_regrets = evaluate(Thompson, horizon, runs)
ola_regrets = evaluate(OptimisticLookAhead, horizon, runs)
gittins_regrets = evaluate(Gittins, horizon, runs)

## Plot dependence on delta

# average the 
shrunkdelta = deltas.reshape(-1, repetitions)[:, 0]
shrunkprobs = np.array([max(p1, p2) for p1, p2 in runs]).reshape(-1, repetitions)[:, 0]


def plot_curve(data, pos, name):
    plt.subplot(1, 4, pos)
    plt.scatter(shrunkdelta, data[:, -1].reshape(-1, repetitions).mean(1) / (shrunkprobs * horizon) * 100, s=10,
                c=shrunkprobs, edgecolors='face', cmap=matplotlib.cm.plasma)
    plt.ylim(-1, 30)
    plt.xlabel('$\Delta$')
    plt.ylabel('Mean propotional regret (\%)')
    plt.title(name)
    plt.grid()


plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plot_curve(ucb_regrets, 1, "UCB")
plot_curve(thompson_regrets, 2, "Thompson")
plot_curve(ola_regrets, 3, "OptimisticLookahed")
plot_curve(gittins_regrets, 4, "Gittins")

plt.savefig('proportional_regret.pdf')
plt.show()
