import numpy as np
import matplotlib.pyplot as plt
from scipy. integrate import odeint
from matplotlib.animation import FuncAnimation
from time import time
from math import sin, cos, pi, sqrt

class sticks:
    def __init__(self,state,m,g,l,k):
        #state=[phi1,dphi1,phi2,dphi2]
        self.state=state
        self.args=[m,l,k,g]

    def compute_positions(self):
        phi1=self.state[0]
        phi2=self.state[2]
        l=self.args[1]
        return [-l*sin(phi1),l*cos(phi1),2*l*sin(phi1),-2*l*cos(phi1),l*sin(phi2),l*cos(phi2),-2*l*sin(phi2),-2*l*cos(phi2)]

    def dstate(self,state,dt):
        m,l,k,g=self.args
        phi1,dphi1,phi2,dphi2=state
        ddphi1=-0.5*(g/l)*sin(phi1) - (k/m)*sin(phi1+phi2)
        ddphi2 = -0.5 * (g / l) * sin(phi2) - (k / m) * sin(phi1 + phi2)
        return [dphi1,ddphi1,dphi2,ddphi2]

    def step(self,dt):
        self.state=odeint(self.dstate,self.state,[0,dt])[1]

    def draw(self,dt):
        fig=plt.figure()
        lim=(-self.args[1]-1,self.args[1]+1)
        ax=fig.add_subplot(111, aspect='equal',autoscale_on=False, xlim=lim,ylim=lim)
        fst, = ax.plot([],[],'k-',lw=3)
        snd, = ax.plot([],[],'k-',lw=3)
        spr, = ax.plot([],[],'b-',lw=1)
        ax.plot([lim[0],lim[1]],[0,0],'r-',lw=0.5)
        ax.plot([0,0],[lim[0],lim[1]],'r-',lw=0.5)

        def init():
            fst.set_data([],[])
            snd.set_data([],[])
            spr.set_data([],[])
            return fst, snd, spr
        def animate(i):
            self.step(dt)
            x1,y1,x2,y2,x3,y3,x4,y4=self.compute_positions()
            fst.set_data([x1,x2],[y1,y2])
            snd.set_data([x3,x4],[y3,y4])
            spr.set_data([x1,x3],[y1,y3])
            return fst, snd, spr

        t0 = time()
        animate(0)
        t1 = time()
        interval = 1000 * dt - (t1 - t0)
        ani = FuncAnimation(fig, animate, frames=300, interval=interval, init_func=init, blit=True)
        plt.show()
if __name__=='__main__':
    st=sticks([1,3,-1,1],1,9.81,1,1.4)
    st.draw(1/60)
