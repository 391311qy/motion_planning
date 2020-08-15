import numpy as np
import numpy.linalg as npl
import scipy

# solve the riccatti equation given following matrix inputs
# A: df(x, u)/dx
# B: df(x, u)/du
# Q: Q
# R: R in J = sum: 0-> inf (xTQx + uTRu)

def lqr(A, B, Q, R):
    # S, K = self.riccattiSolve(A, B, Q, R)
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R)@(B.T@S)
    return S, K

def riccattiSolve(A, B, Q, R):
    # solving ricatti equation efficiently in continuous time
    # ATS + SA - SB(R-1)BTS + Q = 0, or
    # ATS + SA - (SB + N)(R-1)(BTS + NT) + Q = 0, 
    invR = np.linalg.inv(R)
    Z1, Z2, Z3, Z4  = A, - B@invR@B.T, -Q, -A.T
    Z = np.vstack((np.row_stack((Z1, Z2)), np.row_stack((Z3, Z4))))

