''' LQR Control for a Quadrotor using Unit Quaternions:
Modeling and Simulation '''

import numpy as np


########################
#      Quaternion      #
########################
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

def q_to_R(q):
    # http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    R = np.zeros([3,3])
    R[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[0,2] = 2*(q[1]*q[3] + q[0]*q[1])
    R[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[2,1] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,2] =  q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    return R

########################
#    normal methods    #
########################
def R_matrix(z_angle,y_angle,x_angle):
    # s angle: row; y angle: pitch; z angle: yaw
    # generate rotation matrix in SO3
    # RzRyRx = R, ZYX intrinsic rotation
    # also (r1,r2,r3) in R3*3 in {W} frame
    # used in obb.O
    # [[R p]
    # [0T 1]] gives transformation from body to world 
    return np.array([[np.cos(z_angle), -np.sin(z_angle), 0.0], [np.sin(z_angle), np.cos(z_angle), 0.0], [0.0, 0.0, 1.0]])@ \
           np.array([[np.cos(y_angle), 0.0, np.sin(y_angle)], [0.0, 1.0, 0.0], [-np.sin(y_angle), 0.0, np.cos(y_angle)]])@ \
           np.array([[1.0, 0.0, 0.0], [0.0, np.cos(x_angle), -np.sin(x_angle)], [0.0, np.sin(x_angle), np.cos(x_angle)]])

def isinbound(i, x, mode = False, factor = 0, isarray = False):
    if mode == 'obb':
        return isinobb(i, x, isarray)
    if isarray:
        compx = (i[0] - factor <= x[:,0]) & (x[:,0] < i[3] + factor) 
        compy = (i[1] - factor <= x[:,1]) & (x[:,1] < i[4] + factor) 
        compz = (i[2] - factor <= x[:,2]) & (x[:,2] < i[5] + factor) 
        return compx & compy & compz
    else:    
        return i[0] - factor <= x[0] < i[3] + factor and i[1] - factor <= x[1] < i[4] + factor and i[2] - factor <= x[2] < i[5]

def isinobb(i, x, isarray = False):
    # transform the point from {W} to {body}
    if isarray:
        pts = (i.T@np.column_stack((x, np.ones(len(x)))).T).T[:,0:3]
        block = [- i.E[0],- i.E[1],- i.E[2],+ i.E[0],+ i.E[1],+ i.E[2]]
        return isinbound(block, pts, isarray = isarray)
    else:
        pt = i.T@np.append(x,1)
        block = [- i.E[0],- i.E[1],- i.E[2],+ i.E[0],+ i.E[1],+ i.E[2]]
        return isinbound(block, pt)

def lineAABB(p0, p1, dist, aabb):
    # https://www.gamasutra.com/view/feature/131790/simple_intersection_tests_for_games.php?print=1
    # aabb should have the attributes of P, E as center point and extents
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2]  # mid point
    I = [(p1[0] - p0[0]) / dist, (p1[1] - p0[1]) / dist, (p1[2] - p0[2]) / dist]  # unit direction
    hl = dist / 2  # radius
    T = [aabb.P[0] - mid[0], aabb.P[1] - mid[1], aabb.P[2] - mid[2]]
    # do any of the principal axis form a separting axis?
    if abs(T[0]) > (aabb.E[0] + hl * abs(I[0])): return False
    if abs(T[1]) > (aabb.E[1] + hl * abs(I[1])): return False
    if abs(T[2]) > (aabb.E[2] + hl * abs(I[2])): return False
    # I.cross(s axis) ?
    r = aabb.E[1] * abs(I[2]) + aabb.E[2] * abs(I[1])
    if abs(T[1] * I[2] - T[2] * I[1]) > r: return False
    # I.cross(y axis) ?
    r = aabb.E[0] * abs(I[2]) + aabb.E[2] * abs(I[0])
    if abs(T[2] * I[0] - T[0] * I[2]) > r: return False
    # I.cross(z axis) ?
    r = aabb.E[0] * abs(I[1]) + aabb.E[1] * abs(I[0])
    if abs(T[0] * I[1] - T[1] * I[0]) > r: return False

    return True

def lineOBB(p0, p1, dist, obb):
    # transform points to obb frame
    res = obb.T@np.column_stack([np.array([p0,p1]),[1,1]]).T 
    # record old position and set the position to origin
    oldP, obb.P= obb.P, [0,0,0] 
    # calculate segment-AABB testing
    ans = lineAABB(res[0:3,0],res[0:3,1],dist,obb)
    # reset the position
    obb.P = oldP 
    return ans

def OBBOBB(obb1, obb2):
    # https://www.gamasutra.com/view/feature/131790/simple_intersection_tests_for_games.php?print=1
    # each obb class should contain attributes:
    # E: extents along three principle axis in R3
    # P: position of the center axis in R3
    # O: orthornormal basis in R3*3
    a , b = np.array(obb1.E), np.array(obb2.E)
    Pa, Pb = np.array(obb1.P), np.array(obb2.P)
    A , B = np.array(obb1.O), np.array(obb2.O)
    # check if two oriented bounding boxes overlap
    # translation, in parent frame
    v = Pb - Pa
    # translation, in A's frame
    # vdotA[0],vdotA[1],vdotA[2]
    T = [v@B[0], v@B[1], v@B[2]]
    R = np.zeros([3,3])
    for i in range(0,3):
        for k in range(0,3):
            R[i][k] = A[i]@B[k]
            # use separating axis thm for all 15 separating axes
            # if the separating axis cannot be found, then overlap
            # A's basis vector
            for i in range(0,3):
                ra = a[i]
                rb = b[0]*abs(R[i][0]) + b[1]*abs(R[i][1]) + b[2]*abs(R[i][2])
                t = abs(T[i])
                if t > ra + rb:
                    return False
            for k in range(0,3):
                ra = a[0]*abs(R[0][k]) + a[1]*abs(R[1][k]) + a[2]*abs(R[2][k])
                rb = b[k]
                t = abs(T[0]*R[0][k] + T[1]*R[1][k] + T[2]*R[2][k])
                if t > ra + rb:
                    return False

            #9 cross products
            #L = A0 s B0
            ra = a[1]*abs(R[2][0]) + a[2]*abs(R[1][0])
            rb = b[1]*abs(R[0][2]) + b[2]*abs(R[0][1])
            t = abs(T[2]*R[1][0] - T[1]*R[2][0])
            if t > ra + rb:
                return False

            #L = A0 s B1
            ra = a[1]*abs(R[2][1]) + a[2]*abs(R[1][1])
            rb = b[0]*abs(R[0][2]) + b[2]*abs(R[0][0])
            t = abs(T[2]*R[1][1] - T[1]*R[2][1])
            if t > ra + rb:
                return False

            #L = A0 s B2
            ra = a[1]*abs(R[2][2]) + a[2]*abs(R[1][2])
            rb = b[0]*abs(R[0][1]) + b[1]*abs(R[0][0])
            t = abs(T[2]*R[1][2] - T[1]*R[2][2])
            if t > ra + rb:
                return False

            #L = A1 s B0
            ra = a[0]*abs(R[2][0]) + a[2]*abs(R[0][0])
            rb = b[1]*abs(R[1][2]) + b[2]*abs(R[1][1])
            t = abs( T[0]*R[2][0] - T[2]*R[0][0] )
            if t > ra + rb:
                return False

            # L = A1 s B1
            ra = a[0]*abs(R[2][1]) + a[2]*abs(R[0][1])
            rb = b[0]*abs(R[1][2]) + b[2]*abs(R[1][0])
            t = abs( T[0]*R[2][1] - T[2]*R[0][1] )
            if t > ra + rb:
                return False

            #L = A1 s B2
            ra = a[0]*abs(R[2][2]) + a[2]*abs(R[0][2])
            rb = b[0]*abs(R[1][1]) + b[1]*abs(R[1][0])
            t = abs( T[0]*R[2][2] - T[2]*R[0][2] )
            if t > ra + rb:
                return False

            #L = A2 s B0
            ra = a[0]*abs(R[1][0]) + a[1]*abs(R[0][0])
            rb = b[1]*abs(R[2][2]) + b[2]*abs(R[2][1])
            t = abs( T[1]*R[0][0] - T[0]*R[1][0] )
            if t > ra + rb:
                return False

            # L = A2 s B1
            ra = a[0]*abs(R[1][1]) + a[1]*abs(R[0][1])
            rb = b[0] *abs(R[2][2]) + b[2]*abs(R[2][0])
            t = abs( T[1]*R[0][1] - T[0]*R[1][1] )
            if t > ra + rb:
                return False

            #L = A2 s B2
            ra = a[0]*abs(R[1][2]) + a[1]*abs(R[0][2])
            rb = b[0]*abs(R[2][1]) + b[1]*abs(R[2][0])
            t = abs( T[1]*R[0][2] - T[0]*R[1][2] )
            if t > ra + rb:
                return False

            # no separating axis found,
            # the two boxes overlap 
            return True

########################
# quadrotor classes    #
########################
class Quadrotor:

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
        self.F_by_U = np.linalg.inv(self.U_by_F)

        # contorl input range
        self.contorl_max = self.U_by_F@np.array([self.omega_norm[1]**2]*4)


    def linearized_A(self, x0, u0):
        #-------------------quadrotor
        # TODO: refine the motion model. should be linearized at a different point
        A = np.zeros([12,12])
        A[0,1] = 0.5
        A[2,3] = 0.5
        A[4,5] = 0.5
        A[6,7] = 1
        A[7,2] = -2*self.g
        A[8,9] = 1
        A[9,0] = 2*self.g
        A[10,11] = 1
        return A

    def linearized_B(self,x0, u0):
        # B matrix
        # TODO: refine the motion model. should be linearized at a different point
        B = np.zeros([12,4])
        B[1,1] = 1/self.Ixx
        B[3,2] = 1/self.Iyy
        B[5,3] = 1/self.Izz
        B[11,0] = 1/self.m
        return B

    def state_to_OBB(self, x):
        # get an oriented bounding box representing the quadrotor from a state
        pos = (x[6], x[8], x[10])
        q_vec = [x[0], x[2], x[4]]
        q0 = np.sqrt(1 - np.linalg.norm(q_vec))
        q = [q0] + q_vec
        R = q_to_R(q)
        OBB = obb(P = pos, E = (self.l, self.l, 1), O = R)
        return OBB

    
class obb(object):
    # P: center point
    # E: extents
    # O: Rotation matrix in SO(3), in {w}
    def __init__(self, P, E, O):
        self.P = P
        self.E = E
        self.O = O
        self.T = np.vstack([np.column_stack([self.O.T,-self.O.T@self.P]),[0,0,0,1]])

class env:

    def __init__(self, xmin=-10, ymin=-10, zmin=-10, xmax=10, ymax=10, zmax=10):
        self.boundary = np.array([xmin, ymin, zmin, xmax, ymax, zmax]) 
        self.OBB = np.array([obb([0.0,0.0,-5.0],[1.0,1.0,1.0],R_matrix(135,0,0)),
                             obb([3.0,3.0,3.0],[0.5,2.0,2.5],R_matrix(45,0,0))])
        self.quad = Quadrotor()
        self.start = tuple([0]*12)
        self.goal = tuple([1]*12)

    def isinobs(self, x):
        for i in self.OBB:
            if isinbound(i, x[0:3], mode='obb'):
                return True
        return False

    def isCollide(self, child, dist=None):
        '''see if obb specifed by the state intersects obstacle'''
        # x = (q1,w1,q2,w2,q3,w3,x,vx,y,vy,z,vz) in R12
        # construct the OBB formed by the current state
        pos = (child[6], child[8], child[10])
        OBB = self.quad.state_to_OBB(child)
        # check if the position is in bound
        if not isinbound(self.boundary, pos): 
            return True
        # check collision with obb as obstacles
        for i in self.OBB:
            if OBBOBB(OBB, i):
                return True
        return False

    def getDist(self, x, child):
        # get the distance between two states
        return np.sqrt((child[6] - x[6])**2 + (child[8] - x[8])**2 + (child[10] - x[10])**2)

    def sampleFree(self):
        # x = (q1,w1,q2,w2,q3,w3,x,vx,y,vy,z,vz) in R12
        state = [0]*12
        q = self.sampleUnitQuaternion()
        x = self.sampleFreePos()
        w = np.random.uniform(-20, 20, size=3)
        v = np.random.uniform(-10, 10, size=3)
        state[0], state[2], state[4] = q[1], q[2], q[3]
        state[1], state[3], state[5] = w[0], w[1], w[2]
        state[6], state[8], state[10] = x[0], x[1], x[2]
        state[7], state[9], state[11] = v[0], v[1], v[2]
        return tuple(state)

    def sampleUnitQuaternion(self):
        # https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
        q = [0]*4
        s = np.random.uniform()
        sig1 = np.sqrt(1 - s)
        sig2 = np.sqrt(s)
        theta1 = 2 * np.pi * np.random.uniform()
        theta2 = 2 * np.pi * np.random.uniform()
        q[0] = np.cos(theta2) * sig2
        q[1] = np.sin(theta1) * sig1
        q[2] = np.cos(theta1) * sig1
        q[3] = np.sin(theta2) * sig2
        return q

    def sampleFreePos(self):
        x = np.random.uniform(self.boundary[0:3], self.boundary[3:6])
        for i in self.OBB:
            if isinobb(i, x):
                return self.sampleFreePos()
        return x
    
if __name__ == '__main__':
    Env = env()
    q = Env.sampleUnitQuaternion()
    print(np.linalg.norm(q))