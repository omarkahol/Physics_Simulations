import numpy as np
import matplotlib.pyplot as plt 
from scipy. integrate import odeint
from matplotlib.animation import FuncAnimation
from time import time
from math import sin, cos, pi
from mpl_toolkits.mplot3d import Axes3D

class Psys:

    def __init__(self, initState, params, g=9.81):
        self.g=g
        self.state=initState
        self.params=params
        #params=[l,m]
        #state=[phi, dphi, r, dr]
        
    def position(self):
        phi=self.state[0]
        r=self.state[2]

        x=r*cos(phi)
        y=r*sin(phi)
        z=(r**2)/(2*self.params[0])
        return x, y, z
   
    def  dstate(self,state,dt):
        
        dphi=state[1]
        r=state[2]
        dr=state[3]

        l=state[0]
        m=state[1]
        
        denominator=1 + (r**2)/(l**2)
        numerator = dphi**2 - (dr**2)/(l**2) - self.g/l

        ddr=r * numerator/denominator

        ddphi = -(2*dr*dphi)/r

        return [dphi, ddphi, dr, ddr]
    
    def step(self,dt):
        self.state=odeint(self.dstate, self.state, [0, dt])[1]

if __name__=='__main__':
    pd = Psys([np.pi,5,7000,1],[10000,1])
    dt=1/360
    
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d', aspect='equal', autoscale_on=False, xlim=(-10000, 10000), ylim=(-10000, 10000), zlim=(-10000,10000))
    ax.grid()
    line, = ax.plot([], [], [],  'ro', lw=2)
    
    
    l = pd.params[0]
    xs=np.linspace(-10000,10000,700)
    ys=xs
    xs, ys = np.meshgrid(xs,ys)
    zs= ((xs**2) + (ys**2))/(2*l)

    ax.plot_wireframe(xs,ys,zs, lw=0.5)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line, 

    def animate(t):
        global pd, dt
        pd.step(dt)
        x, y, z = pd.position()
        line.set_data([x],[y])
        line.set_3d_properties([z])
        return line,

    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)
    
    ani = FuncAnimation(fig, animate, frames=300, interval=interval, init_func=init, blit=True)
    plt.show()
    
