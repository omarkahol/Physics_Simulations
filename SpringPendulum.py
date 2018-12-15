import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sin, cos
import matplotlib.animation as animation

class SpringPendulum: 
    
    def __init__(self, state, k_m, g=9.81):
        self.state=state
        self.param=k_m
        self.g=g
        #state=[r, dr, theta, dtheta]

    def position(self):
        r=self.state[0]
        theta=self.state[2]

        x=r*sin(theta)
        y=-r*cos(theta)

        return x, y

    def dstate(self, state, dt):
        r=state[0]
        dr=state[1]
        theta=state[2]
        dtheta=state[3]

        ddr = ((dtheta**2) -  self.param)*r + self.g*cos(theta)
        ddtheta = -(self.g/r)*sin(theta) - 2*dr*dtheta/r

        return [dr, ddr, dtheta, ddtheta]

    def step(self, dt):
        self.state=odeint(self.dstate, self.state, [0,dt])[1]
    
    def trajectory(self, t):
        data=odeint(self.dstate, self.state, np.linspace(0,t,1000))
        r=np.array([el[0] for el in data])
        theta=np.array([el[2] for el in data])

        xs=r*np.sin(theta)
        ys=-r*np.cos(theta)
        return xs, ys
        

if __name__=='__main__':
    pd = SpringPendulum([3,-2,np.pi,7], 10)
    dt=1/120

    fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    
    trajectory=pd.trajectory(10)
    ax.plot(trajectory[0], trajectory[1], 'r', lw=1)

    def init():
        line.set_data([], [])
        return line,

    def animate(t):
        global pd, dt
        pd.step(dt)
        x, y = pd.position()
        line.set_data(np.linspace(0,x,2), np.linspace(0,y,2))
        return line,


    from time import time
    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, init_func=init, blit=True)
    plt.show()

