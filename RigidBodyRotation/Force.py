import numpy as np

class Force:
    def __init__(self, origin, name="None"):

        if len(origin)==3:
            self.name = name
            self.origin = np.array(origin, dtype=float)
            self.__origin__ = np.array(origin, dtype=float)
            self.__expr__ = lambda x,y,z,t: np.array([0,0,0])
        else:
            raise ValueError('length must be three')

    def add_expression(self,expr):
        try:
            expr(-1,-1,-1,0)
            self.__expr__=expr
        except:
            raise ValueError("The force must be like lambda x,y,z,t: np.array([Fx,Fy,Fz]). If it constant just write lambda x,y,z,t: np.array([a,b,c])")

    def set_origin(self,origin):
        if len(origin)==3:
            self.origin=origin
            return True
        else:
            return False

    def compute_torque(self,t0):
        return np.cross(self.origin,self.apply_force(t0))


    def apply_force(self,t0):
        if t0 >= 0:
            return np.array(self.__expr__(*self.origin,t0))
        else:
            raise ValueError('Negative time not accepted')




