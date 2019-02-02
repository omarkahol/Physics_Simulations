import numpy as np
from scipy.integrate import odeint
import  matplotlib.pyplot as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Object:

    def __init__(self,I1,I2,I3,b,a,g,w):
        self.IO=[I1,I2,I3]
        self.orient=[b,a,g]
        self.__compute_init_state__(w)

    def r_matrix(self,theta,phi,csi):
        a11 = cos(phi) * cos(csi) - cos(theta) * sin(phi) * sin(csi)
        a12 = -cos(csi) * cos(theta) * sin(phi) - cos(phi) * sin(csi)
        a13 = sin(theta) * sin(phi)

        a21 = cos(csi) * sin(phi) + cos(phi) * cos(theta) * sin(csi)
        a22 = cos(phi) * cos(csi) * cos(theta) - sin(phi) * sin(csi)
        a23 = -cos(phi) * sin(theta)

        a31 = sin(csi) * sin(theta)
        a32 = cos(csi) * sin(theta)
        a33 = cos(theta)

        return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    def __compute_init_state__(self,w):
        rm=self.r_matrix(*self.orient)
        e1=rm.dot([1,0,0])
        e2=rm.dot([0,1,0])
        e3=rm.dot([0,0,1])
        self.state=[*e1,*e2,*e3,*w]

    def w_star(self):
        e1=self.state[0:3]
        e2=self.state[3:6]
        e3=self.state[6:9]
        wp=self.state[9:12]
        return wp[0]*np.array(e1) + wp[1]*np.array(e2) + wp[2]*np.array(e3)
    
    def Mo(self):
        e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, w1p, w2p, w3p = self.state
        I=np.diag(self.IO)
        w=np.array([w1p,w2p,w3p])
        return np.array([[e1x,e2x,e3x],[e1y,e2y,e3y],[e1z,e2z,e3z]]).dot(I.dot(w))

    def dstate(self,state,dt):
        w1 = state[9]
        w2 = state[10]
        w3 = state[11]

        e1 = state[0:3]
        e2 = state[3:6]
        e3 = state[6:9]

        I1, I2, I3 = self.IO

        dw1 = w2 * w3 * (I2 - I3) / I1
        dw2 = w1 * w3 * (I3 - I1) / I2
        dw3 = w1 * w2 * (I1 - I2) / I3

        w=self.w_star()

        de1 = np.cross(w, e1)
        de2 = np.cross(w, e2)
        de3 = np.cross(w, e3)

        return [*de1, *de2, *de3, dw1, dw2, dw3]

    def step(self,dt):
        self.state=odeint(self.dstate,self.state,[0,dt])[-1]

    def draw(self,dt,xs,ys,zs):
        fig=plt.figure()
        lim=(-3,3)
        ax=fig.add_subplot(111,projection='3d',aspect='equal',autoscale_on=False,xlim=lim,ylim=lim,zlim=lim)

        ax.plot([0, 1], [0, 0], [0, 0], 'r-', lw=1)
        ax.plot([0, 0], [0, 1], [0, 0], 'r-', lw=1)
        ax.plot([0, 0], [0, 0], [0, 1], 'r-', lw=1)

        e1, = ax.plot([], [], [], 'b-', lw=2)
        e2, = ax.plot([], [], [], 'b-', lw=2)
        e3, = ax.plot([], [], [], 'b-', lw=2)
        w, = ax.plot([], [], [], 'k-', lw=2)
        M, = ax.plot([],[],[], 'r-', lw=2)

        bd, = ax.plot([],[],[],'g',lw=1)

        a=1/sqrt(self.IO[0])
        b=1/sqrt(self.IO[1])
        c = 1 / sqrt(self.IO[2])

        def animate(i):
            self.step(dt)
            e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, w1p, w2p, w3p = self.state

            w1, w2, w3 = self.w_star()
            e1.set_data([0, e1x], [0, e1y])
            e2.set_data([0, e2x], [0, e2y])
            e3.set_data([0, e3x], [0, e3y])
            w.set_data([0, w1], [0, w2])
            e1.set_3d_properties([0, e1z])
            e2.set_3d_properties([0, e2z])
            e3.set_3d_properties([0, e3z])
            w.set_3d_properties([0, w3])
            
            mom=self.Mo()
            M.set_data([0,mom[0]],[0,mom[1]])
            M.set_3d_properties([0,mom[2]])

            xp=[]
            yp=[]
            zp=[]

            for i in range(0,len(xs)):
                p=np.array([[e1x,e2x,e3x],[e1y,e2y,e3y],[e1z,e2z,e3z]]).dot(np.array([xs[i],ys[i],zs[i]]))
                xp.append(p[0])
                yp.append(p[1])
                zp.append(p[2])

            bd.set_data(xp,yp)
            bd.set_3d_properties(zp)

            return e1,e2,e3,w,bd,M,

        from time import time
        t0 = time()
        animate(0)
        t1 = time()
        inte = 1000 * dt - (t1 - t0)
        ani = FuncAnimation(fig, animate, frames=300, interval=inte, blit=True)
        plt.show(ani)

if __name__=='__main__':

      obj=Object(1,2.5,5,0,0,0,[1e-3,7,0])

      u,v=np.meshgrid(np.linspace(0,pi,10),np.linspace(0,2*pi,20))
      xs=np.hstack(np.sin(u)*np.cos(v))
      ys=np.hstack(np.sin(u)*np.sin(v))
      zs=np.hstack(np.cos(u)/sqrt(3))

      obj.draw(1/120,xs,ys,zs)








