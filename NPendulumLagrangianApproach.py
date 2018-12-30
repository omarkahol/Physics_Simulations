import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy.physics.mechanics import *
import sympy
import math
from sympy import Dummy, lambdify, symbols
from time import time
from matplotlib.animation import FuncAnimation

class NPendum:

    def __init__(self,n,state,lengths,masses,g):
        self.n=n
        self.state=state
        self.lengths=lengths
        self.masses=masses
        self.g=g
        self.args=[g]+lengths+masses
        print('PENDULUM INITIALIZED')
        #state=[position,velocities]

    def extract_equations(self):
        q=dynamicsymbols('q:{0}'.format(self.n))
        u=dynamicsymbols('q:{0}'.format(self.n),1)
        m=symbols('m:{0}'.format(self.n))
        l=symbols('l:{0}'.format(self.n))
        g,t=symbols('g t')

        A=ReferenceFrame('A')
        P=Point('P')

        P.set_vel(A,0)

        BL=[]

        for i in range(self.n):
            Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
            Ai.set_ang_vel(A, u[i] * A.z)

            Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
            Pi.v2pt_theory(P, A, Ai)

            Pai = Particle('Pa' + str(i), Pi, m[i])
            Pai.potential_energy = -sum( [ m[n]*g*l[n]*sympy.cos(q[n]) for n in range(0,i+1) ] )
            BL.append(Pai)
            P=Pi
        print('MODEL CREATED, INITIALIZING LAGRANGIAN CALCULATION...')

        L=Lagrangian(A, *BL)
        l_eqn = LagrangesMethod(L, q)
        l_eqn.form_lagranges_equations()

        print('LAGRANGIAN CALCULATED, EXTRACTING EQUATIONS...')

        parameters = [g] + list(l) + list(m)
        unknowns = [Dummy() for i in q + u]
        unknown_dict = dict(zip(q + u, unknowns))

        mm = l_eqn.mass_matrix_full.subs(unknown_dict)
        fo = l_eqn.forcing_full.subs(unknown_dict)

        self.MM = lambdify(unknowns + parameters, mm)
        self.FO = lambdify(unknowns + parameters, fo)

        print('EQUATIONS EXTRAXTED...')

    def dstate(self,state,dt,args):
        vals = np.concatenate((state, args))
        sol = np.linalg.solve(self.MM(*vals), self.FO(*vals))
        return np.array(sol).T[0]

    def step(self,dt):
        self.state = odeint(self.dstate, self.state, [0, dt], args=(self.args,))[1]

    def position(self):
        q = self.state[0:self.n]
        xi = [0]
        yi = [0]
        for i in range(1, self.n + 1):
            xs = 0
            ys = 0
            for j in range(i):
                xs += self.lengths[j - 1] * math.sin(q[j - 1])
                ys += -self.lengths[j - 1] * math.cos(q[j - 1])
            xi.append(xs)
            yi.append(ys)
        return xi, yi

    def draw(self, dt):
        self.extract_equations()
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))

        line, = ax.plot([], [], 'o-', lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            self.step(dt)
            x, y = self.position()
            line.set_data(x, y)
            return line,

        t0 = time()
        animate(0)
        t1 = time()
        interval = 1000 * dt - (t1 - t0)

        ani = FuncAnimation(fig, animate, frames=300, interval=interval, blit=True, init_func=init)
        plt.show()



if __name__=='__main__':
    n=5
    pos=[1 for i in range(n)]
    vel=[(i**i)%n for i in range(n)]
    l=[1 for i in range(n)]
    m=[1 for i in range(n)]
    p=NPendum(n,pos+vel,l,m,9.81)
    p.draw(1/60)




