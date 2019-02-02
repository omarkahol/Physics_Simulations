import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.integrate import odeint

x0=0
y0=0
xst=-0.5
theta_0=0.1
n1=1
n = lambda y: sqrt(y+1)
yst=-xst/tan(theta_0)

def model(state, t):
    x, y = state
    dy=-1
    dx= tan(asin( (sin(theta_0)*n1) / n(-y) ))
    return [dx, dy]

sol=odeint(model,[x0,y0],np.linspace(0,20,1000))

x=[el[0] for el in sol]
y=[el[1] for el in sol]
fig=plt.figure()
ax=fig.add_subplot(111,autoscale_on=True)
ax.plot([xst,x0],[yst,y0],'r-',lw=2)
ax.plot([min(x+[xst]),max(x)],[y0,y0],'k',lw=1)
ax.plot(x,y,'r',lw=2)
plt.show()
