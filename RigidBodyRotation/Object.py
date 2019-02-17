import numpy as np
from RigidBodyRotation.Force import *
from RigidBodyRotation.ReferenceFrame import *
from RigidBodyRotation.QuatPy import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation

class Object:

    def __init__(self, Io, N, m, name="None"):
        if Io.shape==(3,3):
            self.name = name
            self.N=N
            self.Io=Io
            self.mass=m
            self.omega = np.array([0,0,0])
            self.v = np.array([0,0,0])
            self.__object_time__ = 0
            self.forces=[]
        else:
            raise ValueError('Io must be a 3x3 matrix')

    def apply_w(self,w0):
        if len(w0)==3:
            self.omega=self.N.to_this_frame(w0)
            return True
        else:
            return False

    def add_init_v(self,v):
        if len(v)==3:
            self.v=v
        else:
            return False

    def apply_forces(self,Farr):
        self.forces=Farr

    def __compute_forces_torques__(self):
        R=np.array([0,0,0],dtype=float)
        N=np.array([0,0,0],dtype=float)
        for force in self.forces:
            R += force.apply_force(self.__object_time__)
            N += self.N.to_this_frame(force.compute_torque(self.__object_time__))
        return R, N

    def __compute_state__(self):
        self.__state__ = [*self.N.origin,*self.v,self.N.base_quaternion.s,*self.N.base_quaternion.v,*self.omega]
        print(self.__state__)

    def dtstate(self,state,t):
        #Nx,Ny,Nz,vx,vy,vz,b0,b1,b2,b3,w1,w2,w3,FORCE ORIGINS=state

        R, N = self.__compute_forces_torques__()

        dvx,dvy,dvz = R/self.mass
        w1,w2,w3=state[10:13]
        b0,b1,b2,b3=state[6:10]
        vx,vy,vz=state[3:6]

        dw1,dw2,dw3 = np.linalg.solve(self.Io, np.subtract( N, np.cross( [w1,w2,w3], self.Io.dot([w1,w2,w3]) ) ) )

        dbeta = Quaternion(b0,[b1,b2,b3]).prod(Quaternion(0, [w1,w2,w3])).scale(0.5)
        return_arr=[vx,vy,vz,dvx,dvy,dvz,dbeta.s,*dbeta.v,dw1,dw2,dw3]

        return return_arr

    def ellipsoid(self,n1,n2):
        a = 1 / sqrt(self.Io[0,0])
        b = 1 / sqrt(self.Io[1,1])
        c = 1 / sqrt(self.Io[2,2])
        u, v = np.meshgrid(np.linspace(0, np.pi, n1), np.linspace(0, 2 * np.pi, n2))
        xs = a * np.hstack(np.sin(u) * np.cos(v))
        ys = b * np.hstack(np.sin(u) * np.sin(v))
        zs = c * np.hstack(np.cos(u))
        return np.vstack([xs,ys,zs])

    def step(self,dt):
        self.__state__=odeint(self.dtstate,self.__state__,[0,dt])[-1]
        Nx, Ny, Nz, vx, vy, vz, b0, b1, b2, b3, w1, w2, w3= self.__state__[0:13]

        self.N.origin=np.array([Nx,Ny,Nz],dtype=float)

        self.N.set_base_quaternion(b0,b1,b2,b3)
        self.omega = self.N.to_base_frame([w1,w2,w3])
        self.__object_time__ += dt

        for force in self.forces:
            force.origin += np.array([vx,vy,vz])*dt
            #force.origin += + np.cross(self.omega, np.subtract(force.origin,self.N.origin))*dt

    def draw(self,dt,n1,n2):
        self.__compute_state__()
        fig = plt.figure()
        lim=(-10,10)
        ax = fig.add_subplot(111,projection='3d',xlim=lim,ylim=lim,zlim=lim)
        e1, = ax.plot([], [], 'b-', lw=2)
        e2, = ax.plot([], [], 'b-', lw=2)
        e3, = ax.plot([], [], 'b-', lw=2)
        w, = ax.plot([], [], [], 'k-', lw=2)

        bd, = ax.plot([], [], [], 'g', lw=1)

        def animate(i):
            self.step(dt)

            e1x, e1y, e1z = self.N.move_with_origin(self.N.e1)
            e2x, e2y, e2z = self.N.move_with_origin(self.N.e2)
            e3x, e3y, e3z = self.N.move_with_origin(self.N.e3)

            ox,oy,oz=self.N.origin

            w1,w2,w3=self.N.move_with_origin(self.omega)

            e1.set_data([ox, e1x], [oy, e1y])
            e2.set_data([ox, e2x], [oy, e2y])
            e3.set_data([ox, e3x], [oy, e3y])
            w.set_data([ox, w1], [oy, w2])
            e1.set_3d_properties([oz, e1z])
            e2.set_3d_properties([oz, e2z])
            e3.set_3d_properties([oz, e3z])
            w.set_3d_properties([oz, w3])

            body = self.N.get_axes_array().dot(self.ellipsoid(n1,n2))
            bd.set_data(body[0]+ ox*np.ones(len(body[0])), body[1] + oy*np.ones(len(body[1]))  )
            bd.set_3d_properties(body[2] + oz*np.ones(len(body[2])))

            return e1,e2,e3,w,bd

        from time import time
        t0 = time()
        animate(0)
        t1 = time()
        inte = 1000 * dt - (t1 - t0)
        ani = FuncAnimation(fig, animate, frames=300, interval=inte, blit=True)
        plt.show(ani)

if __name__ == '__main__':

    N=ReferenceFrame('body',[7,7,0],[1,1,1],pi/6)
    Obj = Object(np.diag([1,1,1.1]),N,1,name="Omar")

    Obj.apply_w([5,5,5])
    Obj.add_init_v([0,9,0])

    f1=Force(N.origin,name='gravity')
    R = lambda x,y,z: sqrt((x**2)+(y**2)+(z**2))
    f1.add_expression(lambda x, y, z, t: np.array(  [(-1000*x)/(R(x,y,z)**3),(-1000*y)/(R(x,y,z)**3),(-1000*z)/(R(x,y,z)**3)]  ))

    Obj.apply_forces([f1])

    Obj.draw(1/60,30,30)









