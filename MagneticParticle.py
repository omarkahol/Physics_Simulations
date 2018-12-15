import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from scipy import integrate
from matplotlib.animation import FuncAnimation

class MagneticParticle:

    def __init__(self, B, v, p):
        self.B=B
        self.v=v
        self.p=p
        self.fig=plt.figure()
        self.ax=self.fig.add_subplot(111,projection='3d')
        self.x=[]
        self.y=[]
        self.z=[]

    def magneticModel(self,y,t,B,extra):
        x=y[0:3]
        v=y[3:]
        force=np.cross(v,B)
        return [v[0], v[1], v[2], force[0], force[1], force[2]]

    def computeTrajectory(self):
        sol=integrate.odeint(self.magneticModel, self.p+self.v,  np.linspace(0,10,300), args=(self.B,0))
        print(sol)
        self.x=[el[0] for el in sol]
        self.y=[el[1] for el in sol]
        self.z=[el[2] for el in sol]

    def update(self,t):
        self.ax.plot([self.x[t]], [self.y[t]],[ self.z[t]], 'ro')

    def show(self):
        self.computeTrajectory()
        self.ax.plot(self.x,self.y,self.z,'b')
        
        anim = FuncAnimation(self.fig, self.update, frames=300, interval=1)
        plt.show()

if __name__=='__main__':
    mp=MagneticParticle([0,0,5],[1,0,1],[0,0,0])
    mp.show()
