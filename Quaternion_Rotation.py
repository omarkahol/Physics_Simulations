from typing import List, Any, Union

import numpy as np
from scipy.integrate import odeint
import  matplotlib.pyplot as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Quaternion:
    def __init__(self, s, v):
        self.s = s
        self.v = np.array(v)

    def sum(self, q):
        return Quaternion(self.s + q.s, self.v + q.v)

    def prod(self, q):
        return Quaternion(self.s * q.s - self.v.dot(q.v), self.s * q.v + q.s * self.v + np.cross(self.v, q.v))

    def scale(self, l):
        self.s = self.s * l
        self.v = l * self.v
        return self

    def norm(self):
        return sqrt((self.s ** 2) + (self.v[0] ** 2) + (self.v[1] ** 2) + (self.v[2] ** 2))

    def normalize(self):
        return self.scale(self.norm())

    def conjugate(self):
        return Quaternion(self.s, [-self.v[0], -self.v[1], -self.v[2]])

    def inv(self):
        return self.conjugate().normalize()

    def __str__(self):
        return str(self.s) + ' + ' + str(self.v[0]) + 'i + ' + str(self.v[1]) + 'j + ' + str(self.v[2]) + 'k'

class Rotation:

    def __init__(self,a,theta,Io, w):
        self.state=[cos(theta/2),*sin(theta/2)*np.array(a)/np.linalg.norm(a)]
        print('Roatation quaternion: '+str(self.base_Quaternion(self.state)))
        self.Io=np.array(Io)
        print('Initializing body frame...')
        self.get_frame(self.base_Quaternion(self.state))
        print('adding angular velocity w = '+str(w))
        self.add_w(w)

    def rotate(self, vector, q):
        return np.array(q.prod(Quaternion(0, vector)).prod(q.inv()).v)

    def body_frame(self,v):
        return np.array([self.e1.dot(v),self.e2.dot(v),self.e3.dot(v)])

    def base_frame(self,v):
        return v[0]*self.e1 + v[1]*self.e2 + v[2]*self.e3

    def get_frame(self, beta):
        self.e1 = self.rotate([1, 0, 0], beta)
        self.e2 = self.rotate([0, 1, 0], beta)
        self.e3 = self.rotate([0, 0, 1], beta)

    def add_w(self,w):
        self.state=[*self.state,*self.body_frame(w)]
        print('State [b0,b1,b3,b4,w1f,w2f,w3f]: '+str(self.state))

    def Mo(self):
        w=np.array(self.state[4:7])
        return self.base_frame(np.diag(self.Io).dot(w))

    def base_Quaternion(self,state):
        return Quaternion(state[0],state[1:4])

    def dstate(self,state,dt):
        beta=self.base_Quaternion(state)
        w1,w2,w3=state[4:7]
        dbeta=beta.prod( Quaternion(0,state[4:7]) ).scale(0.5)

        I1, I2, I3 = self.Io

        dw1 = w2 * w3 * (I2 - I3) / I1
        dw2 = w1 * w3 * (I3 - I1) / I2
        dw3 = w1 * w2 * (I1 - I2) / I3

        return [dbeta.s,*dbeta.v,dw1,dw2,dw3]

    def step(self,dt):
        self.state=odeint(self.dstate,self.state,[0,dt])[-1]
        self.get_frame(self.base_Quaternion(self.state))

    def draw_ellipsoid(self,n1,n2,dt):
        a = 1 / sqrt(self.Io[0])
        b = 1 / sqrt(self.Io[1])
        c = 1 / sqrt(self.Io[2])
        u, v = np.meshgrid(np.linspace(0, pi, n1), np.linspace(0, 2 * pi, n2))
        xs = a*np.hstack(np.sin(u) * np.cos(v))
        ys = b*np.hstack(np.sin(u) * np.sin(v))
        zs = c*np.hstack(np.cos(u))
        self.draw_obj(dt,xs,ys,zs)

    def draw_obj(self,dt,xs,ys,zs):
        fig = plt.figure()
        lim=(-2,2)
        ax = fig.add_subplot(111, projection='3d',aspect='equal', autoscale_on=False, xlim=lim, ylim=lim, zlim=lim)
        e1, = ax.plot([], [], 'b-', lw=2)
        e2, = ax.plot([], [], 'b-', lw=2)
        e3, = ax.plot([], [], 'b-', lw=2)
        w, = ax.plot([], [], [], 'k-', lw=2)
        M, = ax.plot([], [], [], 'r-', lw=2)
        ax.plot([0, 1], [0, 0], [0, 0], 'r-', lw=1)
        ax.plot([0, 0], [0, 1], [0, 0], 'r-', lw=1)
        ax.plot([0, 0], [0, 0], [0, 1], 'r-', lw=1)

        bd, = ax.plot([], [], [], 'g', lw=1)

        def animate(i):
            self.step(dt)
            e1x,e1y,e1z = self.e1
            e2x,e2y,e2z = self.e2
            e3x,e3y,e3z = self.e3

            w1,w2,w3 = self.base_frame(self.state[4:7])

            e1.set_data([0, e1x], [0, e1y])
            e2.set_data([0, e2x], [0, e2y])
            e3.set_data([0, e3x], [0, e3y])
            w.set_data([0, w1], [0, w2])
            e1.set_3d_properties([0, e1z])
            e2.set_3d_properties([0, e2z])
            e3.set_3d_properties([0, e3z])
            w.set_3d_properties([0, w3])
            mom = self.Mo()
            M.set_data([0, mom[0]], [0, mom[1]])
            M.set_3d_properties([0, mom[2]])

            body = np.array([[e1x, e2x, e3x], [e1y, e2y, e3y], [e1z, e2z, e3z]]).dot(np.vstack([xs, ys, zs]))

            bd.set_data(body[0], body[1])
            bd.set_3d_properties(body[2])

            return e1,e2,e3,w,M,bd

        from time import time
        t0 = time()
        animate(0)
        t1 = time()
        inte = 1000 * dt - (t1 - t0)
        ani = FuncAnimation(fig, animate, frames=300, interval=inte, blit=True)
        plt.show(ani)

if __name__=='__main__':
    r=Rotation(np.array([1,1,5]), 0, [2,2,1], [3,3,0])
    r.draw_ellipsoid(10,20,1/60)


