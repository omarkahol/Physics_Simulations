import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.integrate import odeint

R=lambda x,y: sqrt(x**2 + y**2)
dRdy=lambda x,y: (y)/sqrt(x**2 + y**2)
nf = lambda r: 0.5/r if r < 1 else 1

def df(f,r,h):
    return (f(r+h)-f(r-h))/(2*h)

def dstatedt(state,t):
    a,x,y=state
    n=nf(R(x,y))
    dx = (1/n)*sin(a)
    dy=(1/n)*cos(a)
    da=-(1/n**2)*df(nf,R(x,y),1e-5)*dRdy(x,y)*sin(a)
    return [da,dx,dy]

fig=plt.figure()
ax=fig.add_subplot(111,aspect='equal')
theta=np.linspace(0,2*pi,1000)
ax.plot(np.sin(theta),np.cos(theta), 'k-', lw=2)

for ys in np.linspace(-0.9,0.9,20):
    sol=odeint(dstatedt,[pi/2,-3,ys],np.linspace(0,10,100000))
    y=[el[2] for el in sol]
    x=[el[1] for el in sol]
    ax.plot(x,y,'r-', lw=1)

plt.show()

