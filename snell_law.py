import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.integrate import odeint

nfunc = lambda y: 0.4+abs(sin(4*y)+2*cos(3*y)) if y < 3  else 1
def df(f,x0,h):
    return (f(x0+h)-f(x0-h))/(2*h)
a0=0.5
n0=1
x0=0
y0=5
state=[a0,x0,y0]

def dstate(state,t):
    a,x,y=state
    n = nfunc(y)
    dx = (1 / n) * sin(a)
    dy = -(1 / n) * cos(a)
    da=(1/n)*df(nfunc,y,1e-5)*dx
    return [da,dx,dy]

t=np.linspace(0,20,100000)
sol=odeint(dstate,state,t)
y=[el[2] for el in sol]
x=[el[1] for el in sol]
plt.plot(x,y)
plt.show()




