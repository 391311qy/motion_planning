import numpy as np
import numpy.linalg as npl
import scipy

# solve the riccatti equation given following matrix inputs
# A: df(x, u)/dx
# B: df(x, u)/du
# Q: Q
# R: R in J = sum: 0-> inf (xTQx + uTRu)

class lqr:

    def __init__(self):
        # records LQR solution at every point
        self.S_set = {}
        self.K_set = {}
        self.Q, self.R = self.Continuous_Cost_Model()

    def Linearized_Motion_Model(self, x0, u0):
        # return A(x0, u0), B(x0, u0)
        return A, B

    def Continuous_Cost_Model(self):
        # return Q, R
        return Q, R

    def lqr_solve(self, A, B, Q, R):
        # S, K = self.riccattiSolve(A, B, Q, R)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R)@(B.T@S)
        return S, K

    def get_sol(self, x):
        # get the stored solution of CARE
        if x not in self.S_set:
            Q, R = self.Q, self.R
            A, B = self.Linearized_Motion_Model(x, 0)
            S, K = self.lqr_solve(A, B, Q, R)
            self.S_set[x] = S
            self.K_set[x] = K
            return S, K
        return self.S_set[x], self.K_set[x]

    def lqr_policy(self, x, x_p):
        S, K = self.get_sol(x)
        x = np.array(x)
        x_p = np.array(x_p)
        x_bar = x_p - x
        policy = -(K@x_bar)
        return policy

    
