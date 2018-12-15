import numpy as np
import matplotlib.pyplot as plt 
from scipy. integrate import odeint
from matplotlib.animation import FuncAnimation
from time import time
from math import sin, cos, pi
from mpl_toolkits.mplot3d import Axes3D

class SpherePendulum:
    
    def __init__(self,initState, params):
        self.params=params
        #params=[LENGTH, MASS]
        
        self.g=9.81

        self.state=initState
        #state=[THETA,OMEGA,PHI,PHI']

    def position(self):
        L=self.params[0]
        x=L*sin(self.state[0])*cos(self.state[2])
        y=L*sin(self.state[0])*sin(self.state[2])
        z=L*cos(self.state[0])

        return x,y,z

    def dstate_dt(self, state, dt):
        theta=state[0]
        omega=state[1]
        phi=state[2]
        dphi=state[3]

        m=self.params[1]
        l=self.params[0]

        domega_dt = (dphi**2)*sin(theta)*cos(theta) + (self.g/l)*sin(theta)
        ddphi_dt= (-2*cos(theta)*dphi*omega)/sin(theta)

        return [omega, domega_dt, dphi, ddphi_dt]

    def step(self, dt):
        self.state=odeint(self.dstate_dt, self.state, [0,dt])[1]
        

if __name__=='__main__':
    pd = SpherePendulum([np.pi-1,0.1,np.pi-1,4],[3,1])
    dt=1/120
    
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d', aspect='equal', autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5), zlim=(-5,5))
    ax.grid()
    line, = ax.plot([], [], [],  'ro-', lw=2)
    
    
    #x2 + y2 + z2 = l2
    xs=np.linspace(-5,5,100)
    ys=xs
    xs, ys = np.meshgrid(xs,ys)
    zs=np.sqrt(9-(xs**2)-(ys**2))
    ax.plot_wireframe(xs,ys,zs, lw=0.5)
    ax.plot_wireframe(xs,ys,-zs, lw=0.5)


    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line, 

    def animate(t):
        global pd, dt
        pd.step(dt)
        x, y, z = pd.position()
        line.set_data(np.linspace(0,x,2), np.linspace(0,y,2))
        line.set_3d_properties(np.linspace(0,z,2))
        return line,

    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)
    
    ani = FuncAnimation(fig, animate, frames=300, interval=interval, init_func=init, blit=True)
    plt.show()
