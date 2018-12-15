import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sin, cos
import matplotlib.animation as animation

class Pendulum:

    def __init__(self, initState, params, g=9.81):
        self.g=g
        self.state=initState
        self.params=params

    def computePosition(self):
        L=self.params[0]
        theta=self.state[0]
        x=L*sin(theta)
        y=-L*cos(theta)
        return x, y

    def dstate_dt(self, state, dt):
        theta=state[0]
        omega=state[1]
        dstate_dt=[omega, -(self.g/self.params[0]) * sin(theta)]
        return dstate_dt
    
    def step(self, dt):
        self.state=odeint(self.dstate_dt, self.state, [0,dt])[1]
        print(self.state)

if __name__=='__main__':
    pd = Pendulum([np.pi/2, 1 ], [3, 2])
    dt=1/30
    
    fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    
    def init():
        line.set_data([], [])
        return line, 

    def animate(t):
        global pd, dt
        pd.step(dt)
        x, y = pd.computePosition()
        line.set_data(np.linspace(0,x,2), np.linspace(0,y,2))
        return line,


    from time import time
    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)
    
    ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, init_func=init, blit=True)
    plt.show()

