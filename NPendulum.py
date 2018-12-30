import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import odeint
from sympy.physics.mechanics import *
from sympy import Dummy, lambdify, symbols
from time import time
from math import sin, cos, pi
from matplotlib.animation import FuncAnimation

class NPendulum:
    def __init__(self, n, state, args):
        #state=[initpos, initvel]
        #args=[g,[lengths],[masses]]
        
        self.n=n
        self.state=state
        self.args=args
    
    def extract_equations(self):
        q=dynamicsymbols('q:{0}'.format(self.n))
        u=dynamicsymbols('u:{0}'.format(self.n))

        m=symbols('m:{0}'.format(self.n))
        l=symbols('l:{0}'.format(self.n))

        g,t=symbols('g,t')

        A = ReferenceFrame('A')
        P = Point('P')

        P.set_vel(A,0)

        BL=[]
        FL=[]
        KD=[]

        for i in range(self.n):
            Ai = A.orientnew('A'+str(i), 'Axis', [q[i], A.z])
            Ai.set_ang_vel(A, u[i]*A.z)

            Pi=P.locatenew('P'+str(i),l[i]*Ai.x)
            Pi.v2pt_theory(P,A,Ai)

            Pai = Particle('Pa'+str(i),Pi,m[i])
            BL.append(Pai)

            FL.append((Pi,m[i]*g*A.x))
            KD.append(q[i].diff(t)-u[i])
            P=Pi
        
        print('MODEL CREATED ...')

        KM=KanesMethod(A,q_ind=q,u_ind=u,kd_eqs=KD)
        fr, frstar = KM.kanes_equations(BL, FL)

        parameters = [g] + list(l) + list(m)
        
        unknowns=[Dummy() for i in q + u]
        unknown_dict = dict(zip(q+u, unknowns))
        kds = KM.kindiffdict()

        mm= KM.mass_matrix_full.subs(kds).subs(unknown_dict)
        fo = KM.forcing_full.subs(kds).subs(unknown_dict)

        self.MM  = lambdify(unknowns + parameters, mm)
        self.FO  = lambdify(unknowns + parameters, fo)
        print('EQUATIONS EXTRAXTED...')

    def dstate(self,state,t,args):
        vals=np.concatenate((state,args))
        sol=np.linalg.solve(self.MM(*vals), self.FO(*vals))
        return np.array(sol).T[0]
    
    def step(self,dt):
        self.state=odeint(self.dstate,self.state,[0,dt], args=(self.args,))[1]
    
    def position(self):
        l=self.args[1:self.n+1]
        q=self.state[0:self.n]

        xi=[0]
        yi=[0]
        xs=0
        ys=0
        for i in range(0,self.n):
            xs+=l[i]*sin(q[i])
            ys+=-l[i]*cos(q[i])
            xi.append(xs)
            yi.append(ys)
        return xi, yi
    
    def draw(self, dt):
        self.extract_equations()
        fig=plt.figure()
        l = self.args[1:self.n + 1]
        lim = (-sum(l)-1, sum(l)+1)
        ax=fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=lim, ylim=lim)

        line, = ax.plot([],[],'o-',lw=2)

        def init():
            line.set_data([],[])
            return line,

        def animate(i):
            self.step(dt)
            x, y=self.position()
            line.set_data(x, y)
            return line,
        
        t0=time()
        animate(0)
        t1=time()
        interval=1000*dt - (t1-t0)

        ani = FuncAnimation(fig, animate, frames=300, interval=interval, blit=True, init_func=init)
        plt.show()

    def integrate_and_draw(self,t):
        print("SETTING UP GRAPH...")
        l = self.args[1:self.n + 1]
        self.extract_equations()
        fig=plt.figure()
        lim=(-sum(l),sum(l))
        ax=fig.add_subplot(111,aspect='equal',autoscale_on=False, xlim=lim, ylim=lim)

        print("CALCULATING MOTION...")
        p=odeint(self.dstate, self.state, t, args=(self.args,))
        print("EXTRACTING COORDINATES")
        x=[]
        y=[]

        for state in p:
            q = state[0:self.n]
            xs=0
            ys=0
            xi=[0]
            yi=[0]
            for i in range(self.n):
                xs += l[i] * sin(q[i])
                ys += -l[i] * cos(q[i])
                xi.append(xs)
                yi.append(ys)
            x.append(xi)
            y.append(yi)
        line, = ax.plot([], [], 'bo-', lw=1)

        print('STARTING ANIMATION')
        def animate(i):
            line.set_data(x[i],y[i])
            return line,

        def init():
            line.set_data([], [])
            return line,

        ani=FuncAnimation(fig, animate, frames=len(t),interval=1000 * t.max() / len(t),blit=True, init_func=init)
        plt.show()

    @staticmethod
    def animate_multiple(pend_arr, dt=1 / 120):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
        line = []

        for i in range(len(pend_arr)):
            l, = ax.plot([], [], 'go-', lw=1)
            line.append(l)
            pend_arr[i].extract_equations()

        def animate(i):
            for l, pd in zip(line, pend_arr):
                pd.step(dt)
                x, y = pd.position()
                l.set_data(x, y)
            return line

        t0 = time()
        animate(0)
        t1 = time()
        interval = 1000 * dt - (t1 - t0)

        ani = FuncAnimation(fig, animate, frames=100, interval=interval, blit=True)
        plt.show()

if __name__ == '__main__':
    n=10
    pos=[(i-n)**i for i in range(n)]
    vel=[(-1)**i for i in range(n)]
    args=[9.81]+[1 for i in range(2*n)]

    pd=NPendulum(n,pos+vel,args)
    pd.integrate_and_draw(np.linspace(0,10,200))
















        
