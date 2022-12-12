from numpy import *
from scipy import sparse


class u_solver:
    def linear(self, mesh_info, bc_info, D, F_info, u, v, p, p_face, dir):
        nx = mesh_info.a
        ny = mesh_info.b
        dx = mesh_info.c
        dy = mesh_info.d
        north = bc_info.a
        east = bc_info.b
        south = bc_info.c
        west = bc_info.d
        phi_n = bc_info.e
        phi_e = bc_info.f
        phi_s = bc_info.g
        phi_w = bc_info.h
        Fn = F_info.a
        Fe = F_info.b
        Fs = F_info.c
        Fw = F_info.d

        # Calculate local s_p
        s_p = zeros((5, ny, nx))
        s_u = zeros((ny, nx))
        if north != 'zeroGradient':  # north face
            s_p[0, ny - 1, :] = -(2 * D[ny - 1, :] - Fn[ny - 1, :])
            s_u[ny - 1, :] = (2 * D[ny - 1, :] - Fn[ny - 1, :]) * phi_n
        if east != 'zeroGradient':  # east face
            s_p[1, :, nx - 1] = -(2 * D[:, nx - 1] - Fe[:, nx - 1])
            s_u[:, nx - 1] += (2 * D[:, nx - 1] - Fe[:, nx - 1]) * phi_e
        if south != 'zeroGradient':
            s_p[2, 0, :] = -(2 * D[0, :] + Fs[0, :])
            s_u[0, :] += (2 * D[0, :] + Fs[0, :]) * phi_s
        if west != 'zeroGradient':
            s_p[3, :, 0] = -(2 * D[:, 0] + Fw[:, 0])
            s_u[:, 0] += (2 * D[:, 0] + Fw[:, 0]) * phi_w
        s_p[4] = s_p[0] + s_p[1] + s_p[2] + s_p[3]
        # pressure gradient
        if dir == 'x':
            for j in range(0, ny, 1):
                for i in range(0, nx, 1):
                    s_u[j, i] += (p_face[3, j, i] - p_face[1, j, i])
        if dir == 'y':
            for j in range(0, ny, 1):
                for i in range(0, nx, 1):
                    s_u[j, i] += (p_face[2, j, i] - p_face[0, j, i])

        # Internal Fields
        a_n = zeros((ny, nx))
        a_e = zeros((ny, nx))
        a_s = zeros((ny, nx))
        a_w = zeros((ny, nx))
        n = ny * nx
        # a_n
        for j in range(0, ny - 1, 1):  # ny
            for i in range(0, nx, 1):  # nx
                a_n[j, i] = (D[j, i] - Fn[j, i] / 2)
        # a_e
        for j in range(0, ny, 1):
            for i in range(0, nx - 1, 1):
                a_e[j, i] = (D[j, i] - Fe[j, i] / 2)
        # a_s
        for j in range(1, ny, 1):
            for i in range(0, nx, 1):
                a_s[j, i] = (D[j, i] + Fs[j, i] / 2)
        # a_w
        for j in range(0, ny, 1):
            for i in range(1, nx, 1):
                a_w[j, i] = (D[j, i] + Fw[j, i] / 2)
        # a_p Internal cells
        a_p = zeros((ny, nx))
        for j in range(1, ny - 1, 1):
            for i in range(1, nx - 1, 1):
                a_p[j, i] = (a_n[j, i] + a_e[j, i] + a_s[j, i] + a_w[j, i] + (Fe[j, i] - Fw[j, i] + Fn[j, i] - Fs[j, i]))/0.7

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
        a_n_l = -ac_linear[0, 0:n - nx]
        a_e_l = -ac_linear[1, 0:n - 1]
        a_s_l = -ac_linear[2, nx:n]
        a_w_l = -ac_linear[3, 1:n]
        a_p_l = a_p.reshape(n)
        A_partial = sparse.diags(a_s_l, -nx) + sparse.diags(a_w_l, -1) + sparse.diags(a_p_l, 0) + \
                    sparse.diags(a_e_l, +1) + sparse.diags(a_n_l, +nx)

        # Make boundary a_p
        boundary_ap = zeros((ny, nx))
        # north
        for j in range(ny - 1, ny, 1):  # ny
            for i in range(0, nx, 1):  # nx
                boundary_ap[j, i] = a_e[j, i] + a_s[j, i] + a_w[j, i] - s_p[4, j, i] + (
                            Fe[j, i] - Fw[j, i] + Fn[j, i] - Fs[j, i])
        # east
        for j in range(0, ny, 1):  # ny
            for i in range(nx - 1, nx, 1):  # nx
                boundary_ap[j, i] = a_n[j, i] + a_s[j, i] + a_w[j, i] - s_p[4, j, i] + (
                            Fe[j, i] - Fw[j, i] + Fn[j, i] - Fs[j, i])
        # south
        for j in range(0, 1, 1):  # ny
            for i in range(0, nx, 1):  # nx
                boundary_ap[j, i] = a_n[j, i] + a_e[j, i] + a_w[j, i] - s_p[4, j, i] + (
                            Fe[j, i] - Fw[j, i] + Fn[j, i] - Fs[j, i])
        # west
        for j in range(0, ny, 1):  # ny
            for i in range(0, 1, 1):  # nx
                boundary_ap[j, i] = a_n[j, i] + a_e[j, i] + a_s[j, i] - s_p[4, j, i] + (
                            Fe[j, i] - Fw[j, i] + Fn[j, i] - Fs[j, i])

        ap_u = (a_p*0.7 + boundary_ap) # not relaxed A_p

        boundary_ap_l = (boundary_ap/0.7).reshape(n)
        boundary_ap_sparse = sparse.diags(boundary_ap_l, 0)

        if dir == 'x':
            for j in range(0, ny, 1):
                for i in range(0, nx, 1):
                    s_u[j, i] += 0.3 * u[j, i] * ap_u[j, i] / 0.7
        if dir == 'y':
            for j in range(0, ny, 1):
                for i in range(0, nx, 1):
                    s_u[j, i] += 0.3 * v[j, i] * ap_u[j, i] / 0.7

        A_boundary = boundary_ap_sparse
        A = A_partial + A_boundary
        A = sparse.lil_matrix(A)
        b = s_u.reshape(n)

        return (A, b, ap_u)
