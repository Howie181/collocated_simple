from numpy import *
from scipy import sparse

class p_solver:
    def linear(self, mesh_info, D, F_info, u, v, p, p_face, ap_u, ap_v):
        nx = mesh_info.a
        ny = mesh_info.b
        dx = mesh_info.c
        dy = mesh_info.d
        Fn = F_info.a
        Fe = F_info.b
        Fs = F_info.c
        Fw = F_info.d

        # Calculate local s_p & s_u - mass discountinuity
        s_p = zeros((5, ny, nx))
        s_u = zeros((ny, nx))
        for j in range(0, ny, 1):
            for i in range(0, nx, 1):
                s_u[j,i] += ((Fw[j,i] - Fe[j,i]) + (Fs[j,i] - Fn[j,i])) #* dx[0,0]

        # Internal Fields
        a_n = zeros((ny, nx))
        a_e = zeros((ny, nx))
        a_s = zeros((ny, nx))
        a_w = zeros((ny, nx))
        n = ny * nx
        # a_n
        for j in range(0, ny - 1, 1):  # ny
            for i in range(0, nx, 1):  # nx
                a_n[j, i] = 2 * 0.7 / (ap_v[j, i] + ap_v[j+1,i]) # / dx[0,0]* dx[0,0]
                #a_n[j, i] = 2 * 0.7 * (1/(ap_u[j, i] +ap_u[j + 1, i]))
        # a_e
        for j in range(0, ny, 1):
            for i in range(0, nx - 1, 1):
                a_e[j, i] = 2 * 0.7 / (ap_u[j, i] + ap_u[j,i+1]) # / dx[0,0]* dx[0,0]
                #a_e[j, i] = 2 * 0.7 * (1 / (ap_u[j, i] + ap_u[j, i+1]))
        # a_s
        for j in range(1, ny, 1):
            for i in range(0, nx, 1):
                a_s[j, i] = 2 * 0.7 / (ap_v[j, i] + ap_v[j-1,i]) # / dx[0,0]* dx[0,0]
                #a_s[j, i] = 2 * 0.7 * (1/(ap_u[j, i] + ap_u[j - 1, i]))
        # a_w
        for j in range(0, ny, 1):
            for i in range(1, nx, 1):
                a_w[j, i] = 2 * 0.7 / (ap_u[j, i] + ap_u[j,i-1]) # / dx[0,0]* dx[0,0]
                #a_w[j, i] = 2 * 0.7 * (1/(ap_u[j, i] + ap_u[j, i - 1]))
        # a_p Internal cells
        a_p = zeros((ny, nx))
        for j in range(1, ny - 1, 1):
            for i in range(1, nx - 1, 1):
                a_p[j, i] = a_n[j, i] + a_e[j, i] + a_s[j, i] + a_w[j, i]

        # Make partial A
        n = ny * nx
        ac_linear = zeros((4, n))
        ac = zeros((4, ny, nx))
        ac[0] = a_n
        ac[1] = a_e
        ac[2] = a_s
        ac[3] = a_w
        for k in range(0, 4, 1):
            ac_linear[k, :] = ac[k, :, :].reshape(n)
        # linearise matrix for sparse matrix
        a_n_l = - ac_linear[0, 0:n - nx]
        a_e_l = - ac_linear[1, 0:n - 1]
        a_s_l = - ac_linear[2, nx:n]
        a_w_l = - ac_linear[3, 1:n]
        a_p_l = a_p.reshape(n)
        A_partial = sparse.diags(a_s_l, -nx) + sparse.diags(a_w_l, -1) + sparse.diags(a_p_l, 0) + \
                    sparse.diags(a_e_l, +1) + sparse.diags(a_n_l, +nx)

        # Make boundary a_p
        boundary_ap = zeros((ny, nx))
        # north
        for j in range(ny - 1, ny, 1):  # ny
            for i in range(0, nx, 1):  # nx
                boundary_ap[j, i] = a_e[j, i] + a_s[j, i] + a_w[j, i] - s_p[4, j, i]
        # east
        for j in range(0, ny, 1):  # ny
            for i in range(nx - 1, nx, 1):  # nx
                boundary_ap[j, i] = a_n[j, i] + a_s[j, i] + a_w[j, i] - s_p[4, j, i]
        # south
        for j in range(0, 1, 1):  # ny
            for i in range(0, nx, 1):  # nx
                boundary_ap[j, i] = a_n[j, i] + a_e[j, i] + a_w[j, i] - s_p[4, j, i]
        # west
        for j in range(0, ny, 1):  # ny
            for i in range(0, 1, 1):  # nx
                boundary_ap[j, i] = a_n[j, i] + a_e[j, i] + a_s[j, i] - s_p[4, j, i]

        ap_complete= a_p + boundary_ap
        boundary_ap_l = boundary_ap.reshape(n)
        boundary_ap = sparse.diags(boundary_ap_l, 0)

        A_boundary = boundary_ap
        A = A_partial + A_boundary
        A = sparse.lil_matrix(A)
        b = s_u.reshape(n)

        return (A, b, ap_complete, s_u)
