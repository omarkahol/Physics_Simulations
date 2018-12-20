import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sin, cos, pi
import matplotlib.animation as animation
from time import time

class DiskSpring:

    def __init__(self, state, params):
        #state=[phi, dphi]
        #params=[m,R,k,g]
        self.state=state
        self.params=params

    def position(self):
        phi = self.state[0]
        R=self.params[1]
        xP=R*cos(phi)
        yP=R*sin(phi)

        return xP, yP
    
    def dstate(self, state, dt):
        phi, dphi = state
        m,R,k,g = self.params
        ddphi= - ((2/3)*(g/R)*cos(phi) + (16/3)*(k/m)*sin(phi))
        return [dphi, ddphi]
    
    def step(self,dt):
        self.state=odeint(self.dstate,self.state,[0,dt])[-1]

if __name__=='__main__':

    ds=DiskSpring([0,10],[1,1,2,9.81])
    dt=1/120
    theta=np.linspace(0,2*pi,1000)

    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal',xlim=(-5,5),ylim=(-5,5))
    ax.grid()
    ax.set_title('DISK ATTACHED TO A SPRING')
    disc, = ax.plot([],[],'r',lw=2)
    spring, = ax.plot([],[],'b-',lw=1)
    
    ax.plot([-5,5],[0,0],'k',lw=3)
    ax.plot([0],[0],'go',lw=4)

    def init():
        disc.set_data([],[])
        spring.set_data([],[])
        return disc, spring, 
    
    def animate(t):
        global ds, dt
        ds.step(dt)
        xP,yP = ds.position()
        xd, yd = [ds.params[1]*np.cos(theta)+xP*np.ones(1000),ds.params[1]*np.sin(theta)+yP*np.ones(1000)]
        disc.set_data(xd,yd)
        xs=2*xP
        ys=2*yP
        spring.set_data([xs, 4*ds.params[1]],[ys,0])

        return disc, spring,
    
    t0=time()
    animate(1)
    t1=time()
    interval=1000*dt - (t1-t0)

    ani = animation.FuncAnimation(fig,animate,init_func=init, frames=300, interval=interval, blit=True)

    plt.show()



