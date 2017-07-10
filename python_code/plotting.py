"""
Generate plots of value functions constructed based on UCB and Gittins index
"""

import numpy as np
import matplotlib
import pandas as pa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})


## Load Data

ucb_valuefunction = {}
valuefunction_csv = pa.read_csv('valuecomputation/ucb_value.csv')
for (t, p, n, v) in zip(valuefunction_csv.Time, valuefunction_csv.Positive, valuefunction_csv.Negative, valuefunction_csv.Value):
    ucb_valuefunction[(t,p,n)] = v

gittins_valuefunction = {}
valuefunction_csv = pa.read_csv('valuecomputation/gittins_value.csv')
for (t, p, n, v) in zip(valuefunction_csv.Time, valuefunction_csv.Positive, valuefunction_csv.Negative, valuefunction_csv.Value):
    gittins_valuefunction[(t,p,n)] = v
    

## Plotting function
    
def plot_value(tlevel, valuefunction, name):
    prob_points = np.linspace(0,1,20)
    ncounts = np.arange(2, tlevel+2+1)
    
    V = []
    
    # iterate over all counts
    for ncount in ncounts:
        prob_values = [ (pos/ncount,valuefunction[(tlevel,pos,ncount-pos)]) for pos in range(1,ncount)]
        predicted = np.interp(prob_points, *zip(*prob_values))
        V.append(predicted)
        
    V = np.array(V)    
    X,Y = np.meshgrid(prob_points,ncounts)
    
    fig = plt.figure(num=2, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111) #, projection='3d')
    ax.contour(X,Y-2,V)
    plt.xlabel("Expected Arm $a$ Success Probability ($\\frac{\\alpha}{\\alpha+\\beta}$)")
    plt.ylabel("Number of Arm $a$ Pulls ($\\alpha + \\beta - 2$)")
    #ax.set_zlabel("Value Function: $\\upsilon^a_{" + str(tlevel) + "}$")
    plt.title(name + " $t=" + str(tlevel) + "$")
    plt.savefig("valuefunction_" + name + "_t" + str(tlevel) + ".pdf")
    plt.show()
    

## Plot the UCB value function

plot_value(tlevel=10, valuefunction=ucb_valuefunction, name="UCB")
plot_value(tlevel=200, valuefunction=ucb_valuefunction, name="UCB")

## Plot the Gittins value function

plot_value(tlevel=10, valuefunction=gittins_valuefunction, name="GittinsIndex")
plot_value(tlevel=200, valuefunction=gittins_valuefunction, name="GittinsIndex")
