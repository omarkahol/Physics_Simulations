import numpy as np
import matplotlib.pyplot as plt
from scipy. integrate import odeint
from matplotlib.animation import FuncAnimation
from time import time
from math import sin, cos, pi, sqrt
from mpl_toolkits.mplot3d import Axes3D

class NewtonianOrbit:

    def __init__(self, state, params):
        #state=(r1,dr1,theta1,dtheta1,r2,dr2,theta2,dtheta2,z1,dz1,z2,dz2]
        #params=[m1,m2,G]
        x1=state[0]*cos(state[2])
        dx1=state[1]*cos(state[2])-state[0]*state[3]*sin(state[2])
        y1=state[0]*sin(state[2])
        dy1=state[1]*sin(state[2])+state[0]*state[3]*cos(state[2])
        x2=state[4]*cos(state[6])
        dx2=state[5]*cos(state[6])-state[4]*state[7]*sin(state[6])
        y2=state[4]*sin(state[6])
        dy2=state[5]*sin(state[6])+state[4]*state[7]*cos(state[6])

        self.state=[x1,dx1,y1,dy1,x2,dx2,y2,dy2,state[8],state[9],state[10],state[11]]
        self.params=params

    def position(self):
        return [self.state[0],self.state[2],self.state[4],self.state[6],self.state[8],self.state[10]]
    
    def dstate(self,state,dt):
        
        m1, m2, g = self.params
        x1,dx1,y1,dy1,x2,dx2,y2,dy2,z1,dz1,z2,dz2=state
        
        r=sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )

        ddx1=g*m2*(x2-x1)/(r**3)
        ddy1=g*m2*(y2-y1)/(r**3)

        ddx2=g*m1*(x1-x2)/(r**3)
        ddy2=g*m1*(y1-y2)/(r**3)

        ddz1=g*m2*(z2-z1)/(r**3)
        ddz2=g*m1*(z1-z2)/(r**3)

        return [dx1,ddx1,dy1,ddy1,dx2,ddx2,dy2,ddy2,dz1,ddz1,dz2,ddz2]
    def step(self,dt):
        self.state=odeint(self.dstate,self.state,[0,dt])[-1]

if __name__=='__main__':
    orb = NewtonianOrbit([0,0,0,0,7,0,np.pi,1.7,   0,0,1,0 ],[1000,1,1])
    dt=1/120

    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d', aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10,10), zlim=(-5,5))
    ax.grid()
    line1, = ax.plot([], [],[], 'ro', lw=2)
    line2, = ax.plot([],[],[],'bo',lw=2)

    def init():
        line1.set_data([],[])
        line1.set_3d_properties([])
        line2.set_data([],[])
        line2.set_3d_properties([])
        return line1, line2,

    def animate(t):
        global orb, dt
        orb.step(dt)
        x1,y1,x2,y2,z1,z2=orb.position()
        line1.set_data(x1,y1)
        line1.set_3d_properties(z1)
        line2.set_data(x2,y2)
        line2.set_3d_properties(z2)
        return line1, line2,


    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)
    ani = FuncAnimation(fig, animate, frames=300, interval=interval, init_func=init, blit=True)
    plt.show()

