''' LQR Control for a Quadrotor using Unit Quaternions:
Modeling and Simulation '''

import numpy as np

def q_mult(q, p):
    # return multiplication between two quaternion
    A = np.zeros((4,4))
    A[0,0] = p[0]
    A[0, 1:4] = (-p[1], -p[2], -p[3])
    A[1:4, 0] = p[1:4]
    A[1:4, 1:4] =  p[0]* np.eye(3) + hat_cross(p[1:4])
    return A@np.array(q)

def q_inv(q):
    # calculate inverse of a quaternion
    q_bar = np.zeros(4)
    q_bar[0] = q[0]
    q_bar[1:4] = -q[1], -q[2], -q[3]
    return q_bar/np.linalg.norm(q)

def hat_cross(a):
    # input a is len 3 vector
    # return a skew-symmetric matrix
    A = np.zeros((3,3))
    A[0,1] = -a[2]
    A[0,2] = a[1]
    A[1,0] = a[2]
    A[1,2] = -a[0]
    A[2,0] = -a[1]
    A[2,1] = a[0]
    return A

def q_A(q): 
    # A(q0)*(q-q0)
    return np.eye(3) - 2 * hat_cross(q[1:4])

def q_to_R(Q):
    # http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    R = np.zeros([3,3])
    R[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[0.2] = 2*(q[1]*q[3] + q[0]*q[1])
    R[1.0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[2,1] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,2] =  q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    return R

class quadrotor:

    def __init__(self):
        # state is p v q w 
        # where 
        # p = (x, y, z)
        # v = (U, V, W)
        # q = (q0, q1, q2, q3) attitude in body frame
        # w = (P, Q,  R) angular velocity in body frame
        # x = (q1,w1,q2,w2,q3,w3,x,vx,y,vy,z,vz) in R12

        # parameters
        self.g = 9.8 # m/s^2
        self.m = 0.52 # kg
        self.JR = 8.66e-7 # rads/s, single rotor moment of inertia
        self.omega_norm = [0,278] # rad/s, range of angular velocity 
        self.Ixx = 6.228e-2 #kgm^2, moment inertia in x-axis
        self.Iyy = 6.225e-2 #kgm^2, moment inertia in y-axis
        self.Izz = 1.121e-2 #kgm^2, moment inertia in z-axis
        self.l = 0.235 #m rotor mass center length
        self.b = 3.13e-5 #Ns^2 lift coefficient
        self.d = 7.5e-7 # drag coeff

        # matrices
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.U_by_F = np.array([[self.b,   self.b,   self.b,   self.b],
                                [0, self.l*self.b, 0, - self.l*self.b],
                                [- self.l*self.b, 0, self.l*self.b, 0],
                                [self.d,  -self.d,   self.d,  -self.d]])


    def linearized_A(self, x0, u0):
        #-------------------quadrotor
        A = np.zeros([12,12])
        A[0,1], A[2,3], A[4,5], 
        A[6,7], A[7,2], A[8,9], 
        A[9,0], A[10,11] = 0.5, 0.5, 0.5, 1, -2*self.g, 1, 2*self.g, 1
        return A

    def linearized_B(self,x0, u0):
        B = np.zeros([12,4])
        B[1,1] = 1/self.Ixx
        B[3,2] = 1/self.Iyy
        B[5,3] = 1/self.Izz
        B[11,0] = 1/self.m
        return B

    

if __name__ == '__main__':
    q = (1,0,0,0)
    print(q_mult(q, q_inv(q)))
    print(q_mult(q_inv(q), q))