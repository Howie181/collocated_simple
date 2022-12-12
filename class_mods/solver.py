from numpy import *
import scipy as sp
from scipy import linalg, sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator
import time
import inspect

class linear_solver:
    def biCGstab_solver(self, A, b, c):
        n=int(A.shape[0])
        A = csc_matrix(A)
        A_new = A+ c*sparse.eye(n,n);   #return A_new
        ilu = splu(A_new)               #return ilu|
        Mx = lambda x: ilu.solve(x)     #return Mx
        M = LinearOperator((n, n), Mx)
        num_iters = 0
        def report(xk):
            nonlocal num_iters
            num_iters += 1
        phi, exitcode = sparse.linalg.bicgstab(A,b, x0=None, tol=1e-08, maxiter=10000, M=M, callback=report)
        #print('exitcode:', exitcode)
        print('number of iterations: %s ,'%num_iters, end ='')
        return(phi)

    def L_norm_res(self, A, phi, b):
        n=int(A.shape[0])
        res=sum(abs(A*(phi)-b))
        L1_norm=sum(abs(A.dot(phi)-b))/n
        L2_norm=sqrt(sum((A.dot(phi)-b)**2)/n)
        print("final global residual:", res)
        # print("Representative residual L_1:", L1_norm)
        # print("Representative residual L_2:", L2_norm)
        return(res, L1_norm, L2_norm)