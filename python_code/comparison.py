#!/bin/python
from basics import *


## Lookead value function


import pandas as pa

ucb_valuefunction = {}
valuefunction_csv = pa.read_csv('valuecomputation/ucb_value.csv')
for (t, p, n, v) in zip(valuefunction_csv.Time, valuefunction_csv.Positive, valuefunction_csv.Negative, valuefunction_csv.Value):
    ucb_valuefunction[(t,p,n)] = v

gittins_valuefunction = {}
valuefunction_csv = pa.read_csv('valuecomputation/gittins_value.csv')
for (t, p, n, v) in zip(valuefunction_csv.Time, valuefunction_csv.Positive, valuefunction_csv.Negative, valuefunction_csv.Value):
    gittins_valuefunction[(t,p,n)] = v


class ValueFunction:
    """
    Use one-step lookahead with a *linearly separable* value function which
    is precomputed for each arm separately
    valuefunction : the value function to be used in the lookahead
    """

    def __init__(self, valuefunction):
        # initialize prior values
        self.Acountpos = 1;
        self.Acountneg = 1;
        self.Bcountpos = 1;
        self.Bcountneg = 1;
        self.valuefunction = valuefunction

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """

        # the pre-computed value function is 0-based! (t=0 is the first time step)
        vApos = self.valuefunction[(t,self.Acountpos+1,self.Acountneg)] + \
                    self.valuefunction[(t,self.Bcountpos,self.Bcountneg)]
        vAneg = self.valuefunction[(t,self.Acountpos,self.Acountneg+1)]  + \
                    self.valuefunction[(t,self.Bcountpos,self.Bcountneg)]  
        vBpos = self.valuefunction[(t,self.Bcountpos+1,self.Bcountneg)] + \
                    self.valuefunction[(t,self.Acountpos,self.Acountneg)]
        vBneg = self.valuefunction[(t,self.Bcountpos,self.Bcountneg+1)] + \
                    self.valuefunction[(t,self.Acountpos,self.Acountneg)]

        pA = self.Acountpos / (self.Acountpos + self.Acountneg)
        qvalueA = pA * (1 + vApos) + (1 - pA) * vAneg
        
        pB = self.Bcountpos / (self.Bcountpos + self.Bcountneg)
        qvalueB = pB * (1 + vBpos) + (1 - pB) * vBneg

        # adjust value function by subtracting the value of no-action
        #qvalueA -= self.valuefunction[(t,self.Acountpos,self.Acountneg)] + \
        #            self.valuefunction[(t,self.Bcountpos,self.Bcountneg)]
        #
        #qvalueB -= self.valuefunction[(t,self.Acountpos,self.Acountneg)] + \
        #            self.valuefunction[(t,self.Bcountpos,self.Bcountneg)]
                    
        if qvalueA > qvalueB:       return 0
        elif qvalueA < qvalueB:     return 1
        else:                       return bernoulli(0.5)

    def update(self, arm, outcome):
        """ Updates the estimate for the arm outcome """
        if arm == 0:
            if outcome == 1:    self.Acountpos += 1
            else:               self.Acountneg += 1
        elif arm == 1:
            if outcome == 1:    self.Bcountpos += 1
            else:               self.Bcountneg += 1
        else: raise RuntimeError("Invalid arm number")


class ValueFunctionLookahead:
    """
    Use multi-step lookahead with a *linearly separable* value function which
    is precomputed for each arm separately
    valuefunction : the value function to be used in the lookahead
    """
    def __init__(self, valuefunction, lookahead_hor = 1, scale = 1.0):
        # initialize prior values
        self.Acountpos = 1
        self.Acountneg = 1
        self.Bcountpos = 1
        self.Bcountneg = 1
        self.lookahead_hor = lookahead_hor
        self.valuefunction = valuefunction
        self.cache = {}
        self.scale = scale

    def _lookahead(self, state, t, steps_left):
        """ Recursive and dumb lookahead 
            The order of elements in state is:
                Acountpos, Acountneg, Bcountpos, Bcountneg 
            Returns: action, value function
        """
        
        # terminate if this is the last step
        if steps_left == 0:
            return -1, self.scale * (self.valuefunction[(t, state[0], state[1])] + \
                                     self.valuefunction[(t, state[2], state[3])])

        # the cache is specific only to the particular run
        if state in self.cache:
            return self.cache[state]
        
        # the pre-computed value function is 0-based! (t=0 is the first time-step)
        vApos = self._lookahead((state[0]+1, state[1], state[2], state[3]), t+1, steps_left-1)[1]
        vAneg = self._lookahead((state[0], state[1]+1, state[2], state[3]), t+1, steps_left-1)[1]
        vBpos = self._lookahead((state[0], state[1], state[2]+1, state[3]), t+1, steps_left-1)[1]
        vBneg = self._lookahead((state[0], state[1], state[2], state[3]+1), t+1, steps_left-1)[1]

        pA = state[0] / (state[0] + state[1])
        qvalueA = pA * (1 + vApos) + (1 - pA) * vAneg
        
        pB = state[2] / (state[2] + state[3])
        qvalueB = pB * (1 + vBpos) + (1 - pB) * vBneg

        if qvalueA > qvalueB:       r = 0, qvalueA
        elif qvalueA < qvalueB:     r = 1, qvalueB
        else:                       r = bernoulli(0.5), qvalueA
        
        # cache the result
        self.cache[state] = r
        return r

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        # change 1-based time to 0-based
        self.cache.clear()
        return self._lookahead((self.Acountpos, self.Acountneg, self.Bcountpos, self.Bcountneg), t-1, self.lookahead_hor)[0]

    def update(self, arm, outcome):
        """ Updates the estimate for the arm outcome """
        if arm == 0:
            if outcome == 1:    self.Acountpos += 1
            else:               self.Acountneg += 1
        elif arm == 1:
            if outcome == 1:    self.Bcountpos += 1
            else:               self.Bcountneg += 1
        else: raise RuntimeError("Invalid arm number")


## Compute and compare the mean regret of various methods

horizon = 290
trials = 2000

ucb_regrets = evaluate(lambda: UCB(2.0), horizon, trials)
vf_ucb_regrets = evaluate(lambda: ValueFunctionLookahead(ucb_valuefunction, 2), horizon, trials)
vf_gittins_regrets = evaluate(lambda: ValueFunctionLookahead(gittins_valuefunction,2), horizon, trials)
thompson_regrets = evaluate(Thompson, horizon, trials)
gittins_regrets = evaluate(Gittins, horizon, trials)

## Plot the mean regret

plt.figure(num=2, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plot_confidence(ucb_regrets, label='UCB')
plot_confidence(thompson_regrets, label='Thompson')
plot_confidence(vf_ucb_regrets, '--', label='ValueFunction UCB')
plot_confidence(vf_gittins_regrets, '--', label='ValueFunction Git')
plot_confidence(gittins_regrets, label='Gittins')
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('Regret')
plt.grid()
plt.savefig('regrets.pdf')
plt.show()


## Compare the regret of solutions with a zero value function

horizon = 200
trials = 500

gittins_regrets = evaluate(Gittins, horizon, trials)
np.random.seed(40); random.seed(0);
vf_regrets1 = evaluate(lambda: ValueFunctionLookaheadStep(1,0.4,0), horizon, trials)
np.random.seed(40); random.seed(0);
vf_regretsM = evaluate(lambda: ValueFunctionLookaheadStep(10,0.4,5), horizon, trials)

# Plot the mean regret
plt.figure(num=2, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(vf_regrets1.mean(0), '-', label='ValueFunction L1')
plt.plot(vf_regretsM.mean(0), '--', label='ValueFunction LM')
plt.plot(gittins_regrets.mean(0), label='Gittins')
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('Regret')
plt.grid()
#plt.savefig('regrets.pdf')
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

## Optimistic Lookahead (old)


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


## Value Function with Steps

class ValueFunctionLookaheadStep:
    """
    Use multi-step lookahead with a *linearly separable* value function which
    is precomputed for each arm separately.
    
    The same action is fixed for multiple steps. This takes advantage of the
    relatively small branching factor when the result is fixed to a single action.
    """
    def __init__(self, lookahead_hor = 1, scale = 1.0, steps_fix_actions = 0):
        # initialize prior values
        self.Acountpos = 1
        self.Acountneg = 1
        self.Bcountpos = 1
        self.Bcountneg = 1
        self.lookahead_hor = lookahead_hor
        self.cache = {}
        self.scale = scale
        self.step_fix_actions = steps_fix_actions

    def _lookahead(self, state, t, steps_left, fixed_action, fixed_action_steps):
        """ Recursive lookahead withc caching and fixed actions for a given number of steps
            The order of elements in state is:
                Acountpos, Acountneg, Bcountpos, Bcountneg 
            steps_left : total number of lookahead steps left
            fixed_action : if the action is fixed
            fixed_action_steps : how many more steps the action is fixed for
            Returns: action, value function
        """
        global valuefunction
        # terminate if this is the last step
        if steps_left == 0:
            return -1, self.scale * (valuefunction[(t, state[0], state[1])] + valuefunction[(t, state[2], state[3])])

        # the cache is specific only to the particular run
        if state in self.cache:
            return self.cache[state]

        if fixed_action_steps <= 0:
            optimize_action = True
            next_fixed_action_steps = self.step_fix_actions
        else:
            optimize_action = False
            next_fixed_action_steps = fixed_action_steps - 1

        if fixed_action == 0 or optimize_action:
            # the pre-computed value function is 0-based! (t=0 is the first time-step)
            vApos = self._lookahead((state[0], state[1]+1, state[2], state[3]), t+1, steps_left-1,0,next_fixed_action_steps)[1]
            vAneg = self._lookahead((state[0]+1, state[1], state[2], state[3]), t+1, steps_left-1,0,next_fixed_action_steps)[1]
            pA = state[0] / (state[0] + state[1])
            qvalueA = pA * (1 + vApos) + (1 - pA) * vAneg
        else:
            qvalueA = - math.inf
        

        if fixed_action == 1 or optimize_action:
            vBpos = self._lookahead((state[0], state[1], state[2]+1, state[3]), t+1, steps_left-1,1,next_fixed_action_steps)[1]
            vBneg = self._lookahead((state[0], state[1], state[2], state[3]+1), t+1, steps_left-1,1,next_fixed_action_steps)[1]
            pB = state[2] / (state[2] + state[3])
            qvalueB = pB * (1 + vBpos) + (1 - pB) * vBneg
        else:
            qvalueB = - math.inf
            
        assert max(qvalueA, qvalueB) > - math.inf

        if qvalueA > qvalueB:       r = 0, qvalueA
        elif qvalueA < qvalueB:     r = 1, qvalueB
        else:                       r = 0, qvalueA #r = bernoulli(0.5), qvalueA
        
        # cache the result
        self.cache[state] = r
        return r

    def choose(self, t):
        """ Which arm to choose; t is the current time step. Returns arm index """
        # change 1-based time to 0-based
        self.cache.clear()
        return self._lookahead((self.Acountpos, self.Acountneg, self.Bcountpos, self.Bcountneg), t-1, self.lookahead_hor, -1, 0)[0]

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

