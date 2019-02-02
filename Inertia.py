import numpy as np
from scipy.integrate import odeint
import  matplotlib.pyplot as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Object:

    def __init__(self,theta,phi,csi,I1,I2,I3,w):
        self.orient=[theta,phi,csi]
        self.IO=[I1,I2,I3]
        self.omega=w

    def r_matrix(self):
        theta, phi, csi = self.orient

        a11=cos(phi)*cos(csi) - cos(theta)*sin(phi)*sin(csi)
        a12=-cos(csi)*cos(theta)*sin(phi)-cos(phi)*sin(csi) 
        a13=sin(theta)*sin(phi)

        a21=cos(csi)*sin(phi)+cos(phi)*cos(theta)*sin(csi)
        a22=cos(phi)*cos(csi)*cos(theta)-sin(phi)*sin(csi)
        a23= -cos(phi)*sin(theta)
        
        a31=sin(csi)*sin(theta)
        a32=cos(csi)*sin(theta)
        a33=cos(theta)

        return np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
    
    def obtain_base(self):
        mat=self.r_matrix()
        self.e1=mat.dot(np.array([1,0,0]))
        self.e2=mat.dot(np.array([0,1,0]))
        self.e3=mat.dot(np.array([0,0,1]))

    def get_omega(self):
        e1 = self.state[0:3]
        e2 = self.state[3:6]
        e3 = self.state[6:9]
        wp = self.state[9:12]
        return wp[0] * np.array(e1) + wp[1] * np.array(e2) + wp[2] * np.array(e3)

    def compute_state(self):
        self.state=[*self.e1,*self.e2,*self.e3,*self.omega]

    def dstate(self,state,dt):
        w1=state[9]
        w2=state[10]
        w3=state[11]

        e1=state[0:3]
        e2=state[3:6]
        e3=state[6:9]

        I1,I2,I3 = self.IO

        dw1=w2*w3*(I2-I3)/I1
        dw2=w1*w3*(I3-I1)/I2
        dw3=w1*w2*(I1-I2)/I3

        w_star=self.get_omega()

        de1=np.cross(w_star,e1)
        de2=np.cross(w_star,e2)
        de3=np.cross(w_star,e3)

        return [*de1,*de2,*de3,dw1,dw2,dw3]
    
    def step(self,dt):
        self.state=odeint(self.dstate,self.state,[0,dt])[-1]
        
    
    def draw(self,dt):
        fig=plt.figure()
        lim=(-3,3)
        ax=fig.add_subplot(111,projection='3d', autoscale_on=False, aspect='equal', xlim=lim, ylim=lim,zlim=lim)
        ax.plot([0,1],[0,0],[0,0],'r-',lw=1)
        ax.plot([0,0],[0,1],[0,0],'r-',lw=1)
        ax.plot([0,0],[0,0],[0,1],'r-',lw=1)

        e1, = ax.plot([],[],[],'b-',lw=1)
        e2, = ax.plot([],[],[],'b-',lw=1)
        e3, = ax.plot([],[],[],'b-',lw=1)
        w, = ax.plot([],[],[], 'k-', lw=1)

        b1,= ax.plot([],[],[], 'g-',lw=2)

        i1,i2,i3 = self.IO

        self.obtain_base()
        self.compute_state()

        def animate(i):
            self.step(dt)
            e1x,e1y,e1z,e2x,e2y,e2z,e3x,e3y,e3z, w1p,w2p,w3p = self.state

            p1x, p1y, p1z  = (1/sqrt(i1)) * np.array([e1x,e1y,e1z])
            p2x, p2y, p2z  = (1/sqrt(i2)) * np.array([e2x,e2y,e2z])
            p3x, p3y, p3z  = (1/sqrt(i3)) * np.array([e3x,e3y,e3z])
            
            b1.set_data([p1x,p2x,p3x,-p1x,-p2x,-p3x,-p1x,p2x,-p3x, p1x,p2x,-p3x, p1x,-p2x,p3x,-p1x,p2x,p3x,p1x,-p2x,-p3x],[p1y,p2y,p3y,-p1y,-p2y,-p3y,-p1y,p2y,-p3y,p1y,p2y,-p3y,p1y,-p2y,p3y,-p1y,p2y,p3y,p1y,-p2y,-p3y])
            b1.set_3d_properties([p1z,p2z,p3z,-p1z,-p2z,-p3z,-p1z,p2z,-p3z,p1z,p2z,-p3z,p1z,-p2z,p3z,-p1z,p2z,p3z,p1z,-p2z,-p3z])

            w1,w2,w3 = self.get_omega()
            e1.set_data([0,e1x],[0,e1y])
            e2.set_data([0,e2x],[0,e2y])
            e3.set_data([0,e3x],[0,e3y])
            w.set_data([0,w1],[0,w2])
            e1.set_3d_properties([0,e1z])
            e2.set_3d_properties([0,e2z])
            e3.set_3d_properties([0,e3z])
            w.set_3d_properties([0,w3])
            return e1,e2,e3,w,b1

        from time import time
        t0=time()
        animate(0)
        t1=time()
        inte=1000*dt - (t1-t0)
        ani = FuncAnimation(fig,animate,frames=300,interval=inte, blit=True)
        plt.show()
if __name__=='__main__':
    obj = Object(0,0,0, 1,2,5, [0,5,1e-3])
    obj.draw(1/60)





        
