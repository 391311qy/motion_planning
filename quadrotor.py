''' LQR Control for a Quadrotor using Unit Quaternions:
Modeling and Simulation '''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import proj3d

########################
#      Quaternion      #
########################
def q_mult(q, p):
    # return multiplication between two quaternion
    return q_L(q)@np.array(p)

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

def q_L(q):
    return np.array(
        [[q[0], -q[1], -q[2], -q[3]],
         [q[1],  q[0], -q[3],  q[2]],
         [q[2],  q[3],  q[0], -q[1]],
         [q[3], -q[2],  q[1],  q[0]]])

def q_R(q):
    return np.array(
        [[q[0], -q[1], -q[2], -q[3]],
         [q[1],  q[0],  q[3], -q[2]],
         [q[2], -q[3],  q[0],  q[1]],
         [q[3],  q[2], -q[1],  q[0]]])

def q_to_R(q, homogeneous = True):
    # http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    if homogeneous:
        '''this is the homogeneous expression of R'''
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
    else:
        '''this is the nonhomogeneous expression of R'''
        R = np.zeros([3,3])
        R[0,0] = 1 - 2 * (q[2]**2 + q[3]**2)
        R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
        R[0,2] = 2*(q[1]*q[3] + q[0]*q[1])
        R[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
        R[1,1] = 1 - 2 * (q[1]**2 + q[3]**2)
        R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
        R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
        R[2,1] = 2*(q[2]*q[3] - q[0]*q[1])
        R[2,2] = 1 - 2 * (q[1]**2 + q[2]**2) 
    return R

def q_to_angles(q):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    thetax = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]) , (1 - 2*(q[1]**2 + q[2]**2)))
    thetay = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
    thetaz = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]) , (1 - 2*(q[2]**2 + q[3]**2)))
    # thetax = np.arctan(2*(q[0]*q[1]+q[2]*q[3]) / (1 - 2*(q[1]**2 + q[2]**2)))
    # thetay = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
    # thetaz = np.arctan(2*(q[0]*q[3]+q[1]*q[2]) / (1 - 2*(q[2]**2 + q[3]**2)))
    return thetax, thetay, thetaz

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
#    visualization     #
########################

def visualization(initparams):
    quad = Quadrotor()
    if initparams.ind % 100 == 0 or initparams.done:
        #----------- list structure
        V = np.array(list(initparams.V))
        edges = np.array([[(i[0][0],i[0][1],i[0][2]),(i[1][0],i[1][1],i[1][2])] for i in initparams.E])
        # E = initparams.E
        #----------- end
        stateOBBlist = np.array([quad.state_to_OBB(v) for v in initparams.V])
        # edges = initparams.E
        Path = np.array(initparams.Path)
        start = initparams.env.start
        ax = plt.subplot(111, projection='3d')
        ax.view_init(elev=60., azim=60.)
        # ax.view_init(elev=-8., azim=180)
        ax.clear()
        # drawing objects
        # draw_Spheres(ax, initparams.env.balls)
        # draw_block_list(ax, initparams.env.blocks)
        if initparams.env.OBB is not None:
            draw_obb(ax, initparams.env.OBB)
        draw_obb(ax, stateOBBlist, color = 'b')
        draw_block_list(ax, np.array([initparams.env.boundary]), alpha=0)
        draw_line(ax, edges, visibility=0.75, color='g')
        draw_line(ax, Path, color='r')
        if len(V) > 0:
            ax.scatter3D(V[:, 0], V[:, 1], V[:, 2], s=2, color='g', )
        # ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
        # ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k')
        # adjust the aspect ratio
        ax.dist = 7
        set_axes_equal(ax)
        make_transparent(ax)
        ax.set_axis_off()
        plt.pause(0.0001)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def make_transparent(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

def draw_line(ax, SET, visibility=1, color=None):
    if SET != []:
        for i in SET:
            xs = i[0][0], i[1][0]
            ys = i[0][1], i[1][1]
            zs = i[0][2], i[1][2]
            line = plt3d.art3d.Line3D(xs, ys, zs, alpha=visibility, color=color)
            ax.add_line(line)

def draw_block_list(ax, blocks, color=None, alpha=0.15):
    '''
    drawing the blocks on the graph
    '''
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                 dtype='float')
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    n = blocks.shape[0]
    d = blocks[:, 3:6] - blocks[:, :3]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = v * d[k] + blocks[k, :3]
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=alpha, linewidths=1, edgecolors='k')
        pc.set_facecolor(color)
        h = ax.add_collection3d(pc)
        return h

def draw_obb(ax, OBB, color=None, alpha=0.15):
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    n = OBB.shape[0]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = obb_verts(OBB[k])
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=alpha, linewidths=1, edgecolors='k')
        pc.set_facecolor(color)
        h = ax.add_collection3d(pc)
        return h

def obb_verts(obb):
    # 0.017004013061523438 for 1000 iters
    ori_body = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], \
                         [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]])
    # P + (ori * E)
    ori_body = np.multiply(ori_body, obb.E)
    # obb.O is orthornormal basis in {W}, aka rotation matrix in SO(3)
    verts = (obb.O @ ori_body.T).T + obb.P
    return verts

########################
# quadrotor classes    #
########################
class Quadrotor:
    '''Foehn, Philipp, and Davide Scaramuzza. 
    "Onboard state dependent lqr for agile quadrotors." 
    2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018.'''

    def __init__(self):
        # x = (x,y,z,q0,q1,q2,q3,vx,vy,vz) : state vector in R10
        # u = (wx, wy, wz, c) : w. is the angular acceleration, c is thrust

        # parameters
        self.g = 9.8 # m/s^2
        self.m = 0.52 # kg
        self.l = 0.2 #m arm length
        self.h = 0.1 #m height of the rotor
        self.w_range = [0, 2] # 0 to 2 radians/s control input can get
        self.c_range = [2, 18] # m/s^-2 limit of thrust in vertical direction

    def linearized_A(self, x0, u0):
        # linearized A at a point
        q = x0[3:7]
        w = x0[7:10]
        c = u0[3]
        A = np.zeros([10, 10])
        dpdv = self.diff_p_dv()
        dqdq = self.diff_q_dq(q, w)
        dvdq = self.diff_v_dq(q, c)
        A[0:3, 7:10] = dpdv
        A[3:7, 3:7] = dqdq
        A[7:10, 3:7] = dvdq
        return A

    def linearized_B(self,x0, u0):
        # linearized B at a point
        q = x0[3:7]
        w = x0[7:10]
        c = u0[3]
        dvdc = self.diff_v_dc(q, c)
        dqdw = self.diff_q_dw(q, w)
        B = np.zeros([10, 4])
        B[3:7, 0:3] = dqdw
        B[7:10, 3] = dvdc
        return B
        
    def diff_unit_quaternion(self, q):
        # differentiation wrt a unit quaternion
        n = np.linalg.norm(q)
        return (np.eye(4) - np.outer(q, q) / n**2) / n

    def diff_p_dv(self):
        # diffrentiation of pdot wrt q
        return np.eye(3)

    def diff_q_dq(self, q, w):
        # diffrentiation of qdot wrt q
        w_comp = [0] + list(w)
        return 0.5 * q_R(w_comp) @ self.diff_unit_quaternion(q)

    def diff_q_dw(self, q, w):
        # differentiation of qdot wrt w
        return 0.5 * np.array([
            [-q[1], -q[2], -q[3]],
            [ q[0], -q[3], -q[2]],
            [ q[3],  q[0],  q[1]],
            [-q[2],  q[1],  q[0]]
        ])

    def diff_v_dq(self, q, c):
        # differentiation of vdot wrt q
        return 2 * c * np.array([
            [ q[2],  q[3],  q[0],  q[1]],
            [-q[1], -q[0],  q[3],  q[2]],
            [ q[0], -q[1], -q[2], -q[3]]
        ]) @ self.diff_unit_quaternion(q)

    def diff_v_dc(self, q, c):
        # differentiation of vdot wrt c
        return np.array([
            q[0] * q[2] + q[1] * q[3],
            q[2] * q[3] - q[0] * q[1],
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        ])

    def state_to_OBB(self, x):
        # get an oriented bounding box representing the quadrotor from a state
        q = x[3:7]
        p = x[0:3]
        q = q / np.linalg.norm(q) # normalization
        R = q_to_R(q)
        # x, y, z = q_to_angles(q)
        # R = R_matrix(z_angle = z, y_angle = y, x_angle = x)
        OBB = obb(P = p, E = (self.l, self.l, self.h), O = R)
        return OBB

    def control_restriction(self, pi):
        # given a contorl input, restrict it
        for i in range(0,3):
            if pi[i] < self.w_range[0]: pi[i] = self.w_range[0]
            elif pi[i] > self.w_range[1]: pi[i] = self.w_range[1]
        if pi[3] < self.c_range[0]: pi[3] = self.c_range[0]
        elif pi[3] > self.c_range[1]: pi[3] = self.c_range[1]
        return pi

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
        q = self.sampleUnitQuaternion()
        self.start = tuple([0.1,0.1,0.1,1,0,0,0,0,0,0])
        self.goal = tuple([-3.0,-3.0,-3.0,q[0],q[1],q[2],q[3],0,0,0])

    def isinobs(self, x):
        for i in self.OBB:
            if isinbound(i, x[0:3], mode='obb'):
                return True
        return False

    def isCollide(self, child, dist=None):
        '''see if obb specifed by the state intersects obstacle'''
        # construct the OBB formed by the current state
        OBB = self.quad.state_to_OBB(child)
        # check collision with obb as obstacles
        pos = child[0:3]
        if not isinbound(self.boundary, pos):
            return True
        for i in self.OBB:
            if OBBOBB(OBB, i):
                return True
        return False

    def getDist(self, x, child):
        # get the distance between two states
        return np.sqrt((child[6] - x[6])**2 + (child[8] - x[8])**2 + (child[10] - x[10])**2)

    def sampleFree(self):
        # x = (x,y,z,q0,q1,q2,q3,vx,vy,vz) in R10
        state = [0]*10
        q = self.sampleUnitQuaternion()
        x = self.sampleFreePos(R = q_to_R(q))
        v = np.random.uniform(-10, 10, size=3)
        state[0], state[1], state[2] = x[0], x[1], x[2]
        state[3], state[4], state[5], state[6] = q[0], q[1], q[2], q[3]
        state[7], state[8], state[9] = v[0], v[1], v[2]
        return tuple(state)

    def sampleUnitQuaternion(self):
        # sample an unit quaternion
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

    def sampleFreePos(self, R):
        # given rotation matrix, construct an obb and see if obb is free of other obbs
        x = np.random.uniform(self.boundary[0:3], self.boundary[3:6])
        OBB = obb(P = x, E = [self.quad.l, self.quad.l, self.quad.h], O = R)
        for i in self.OBB:
            if OBBOBB(i, OBB):
                return self.sampleFreePos(R)
        return x

    
if __name__ == '__main__':
    Env = env()
    q = Env.sampleUnitQuaternion()
    R = q_to_R(q, homogeneous= False)
    x, y, z = q_to_angles(q)
    print((x,y,z))
    R2 = np.linalg.inv(R_matrix(x_angle = x, y_angle = y, z_angle = z))
    print(R) 
    print(R2)