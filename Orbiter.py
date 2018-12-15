import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sin, cos
import matplotlib.animation as animation

class Orbiter:
    def __init__(self, state, params, G=1e-5):
        #state=[r1,dr1,theta1,dtheta1,r2,dr2,theta2,dtheta2]
        self.state=np.array(state)
        #params=[m1,m2]
        self.params=params
        self.G=G
        print('r1= '+str(self.state[0])+' r2= '+str(self.state[4]))
        self.initEnergy=self.energy()

    def position(self):
        r1=self.state[0]
        theta1=self.state[2]
        r2=self.state[4]
        theta2=self.state[6]

        x1=r1*cos(theta1)
        y1=r1*sin(theta2)

        x2=r2*cos(theta2)
        y2=r2*sin(theta2)

        return x1, y1, x2, y2

    def energy(self):
        r1=self.state[0]
        dr1=self.state[1]
        theta1=self.state[2]
        dtheta1=self.state[3]
        r2=self.state[4]
        dr2=self.state[5]
        theta2=self.state[6]
        dtheta2=self.state[7]
        m1=self.params[0]
        m2=self.params[1]
        T1=0.5*m1*(dr1**2 + (r1**2)*(dtheta1**2)) 
        T2=0.5*m2*(dr2**2 + (r2**2)*(dtheta2**2))
        V=2*self.G*m1*m2/(r2-r1)
        return T1+T2+V

    def dstate(self, state, dt):
        r1=self.state[0]
        dr1=self.state[1]
        theta1=self.state[2]
        dtheta1=self.state[3]
        r2=self.state[4]
        dr2=self.state[5]
        theta2=self.state[6]
        dtheta2=self.state[7]

        m1=self.params[0]
        m2=self.params[1]

        ddr1=r1*(dtheta1**2) - (2*self.G*m2)/((r2-r1)**2)
        ddr2=r2*(dtheta2**2) - (2*self.G*m1)/((r2-r1)**2)

        ddtheta1=(-2*dr1*dtheta1)/r1
        ddtheta2=(-2*dr2*dtheta2)/r2

        return [dr1,ddr1,dtheta1,ddtheta1,dr2,ddr2,dtheta2,ddtheta2]

    def step(self,dt):
        self.state=odeint(self.dstate, self.state, [0,dt])[1]
        print(self.energy())

if __name__=='__main__':
    pd = Orbiter([1000,0,np.pi,0,5000,5,0,5],[1e+12,1],1)
    dt=1/360

    fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10000, 10000), ylim=(-10000,10000))
    ax.grid()
    line1, = ax.plot([], [], 'ro', lw=2)
    line2, = ax.plot([],[],'bo',lw=2)

    def init():
        line1.set_data([],[])
        line2.set_data([],[])
        return line1, line2,

    def animate(t):
        global pd, dt
        pd.step(dt)
        x1,y1,x2,y2=pd.position()
        line1.set_data(x1,y1)
        line2.set_data(x2,y2)
        return line1, line2,


    from time import time
    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, init_func=init, blit=True)
    plt.show()
