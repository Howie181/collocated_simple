## Functions are stores here
from functions import *
from class_mods.storage import *
from class_mods.pre_processing import *
from class_mods.solver import *
from class_mods.u_solver import *
from class_mods.p_solver import *
from class_mods.post_processing import *



mesher=meshing()
solver=linear_solver()
momentum_solver=u_solver()
p_sol=p_solver()
pp=post_proc()

def initialization_temp(mesh_info, u_bc_info, v_bc_info, u, v, gam, p):
    nx = mesh_info.a
    ny = mesh_info.b
    dx = mesh_info.c
    D = zeros((ny, nx))
    D = gam / dx

    p = np.zeros((ny, nx))
    p_face = np.zeros((4, ny, nx))
    ap_u = np.zeros((ny, nx))
    ap_u[:, :] = 1
    ap_v = np.zeros((ny,nx))
    ap_v[:, :] = 1
    F_info = make_Fs(u, v, nx, ny, p, p_face, ap_u, ap_v)
    A, b, ap_u = momentum_solver.linear(mesh_info, u_bc_info, D, F_info, u, v, p, p_face, 'x')
    A, b, ap_v = momentum_solver.linear(mesh_info, v_bc_info, D, F_info, u, v, p, p_face, 'y')

    return (D, ap_u, ap_v, F_info)

def p_cor(p, p_prime, ny, nx, alpha_p):
    p_ref=p_prime[0,0]
    p_prime=p_prime-p_ref
    p=p+alpha_p*(p_prime)
    return (p, p_prime)

def u_cor(u_star, v_star, p_prime_face, ap_u, ap_v, ny, nx, alpha_u):
    u_prime=np.zeros((ny,nx))
    u_prime[:,:]=(p_prime_face[3,:,:]-p_prime_face[1,:,:])/ap_u[:,:]

    v_prime=np.zeros((ny,nx))
    v_prime[:,:]=(p_prime_face[2,:,:]-p_prime_face[0,:,:])/ap_v[:,:]

    u_new=u_star+alpha_u*u_prime
    v_new=v_star+alpha_u*v_prime
    return (u_new, v_new)

def solve_u(mesh_info, bc_info, D, F_info, c, u, v, p, p_face):
    nx = mesh_info.a
    ny = mesh_info.b
    print('Solving x-momentum,', end ='')
    A, b, ap_u = momentum_solver.linear(mesh_info, bc_info, D, F_info, u, v, p, p_face, 'x')
    phi = solver.biCGstab_solver(A, b, c)
    res = solver.L_norm_res(A, phi, b)
    phi_reshaped = phi.reshape((ny, nx))
    return (phi_reshaped, ap_u)

def solve_v(mesh_info, bc_info, D, F_info, c, u, v, p, p_face):
    nx = mesh_info.a
    ny = mesh_info.b
    print('Solving y-momentum,', end ='')
    A, b, ap_v = momentum_solver.linear(mesh_info, bc_info, D, F_info, u, v, p, p_face, 'y')
    phi=solver.biCGstab_solver(A, b, c)
    res=solver.L_norm_res(A, phi, b)
    phi_reshaped = phi.reshape((ny, nx))
    return (phi_reshaped, ap_v)

def solve_p(mesh_info, D, F_info, c, u, v, p, p_face, ap_u, ap_v):
    nx = mesh_info.a
    ny = mesh_info.b
    print('Solving pressure field,', end ='')
    A, b, ap_p, s_u = p_sol.linear(mesh_info, D, F_info, u, v, p, p_face, ap_u, ap_v)
    phi=solver.biCGstab_solver(A, b, c)
    res=solver.L_norm_res(A, phi, b)
    phi_reshaped = phi.reshape((ny,nx))
    return (phi_reshaped)


def make_Fs(u, v, nx, ny, p, p_face, ap_u, ap_v):
    Fn = np.zeros((ny, nx))
    Fe = np.zeros((ny, nx))
    Fs = np.zeros((ny, nx))
    Fw = np.zeros((ny, nx))
    # Fn
    for j in range(0, ny - 1, 1):  # ny
        for i in range(0, nx, 1):  # nx
            Fn[j, i] = (v[j, i] + v[j + 1, i]) / 2 + 0.7 *(
                        (p_face[0, j + 1, i] - p_face[0, j, i]) / ap_v[j + 1, i] + (p_face[0, j, i] - p_face[2, j, i]) /
                        ap_v[j, i]) / 2 \
                       - 2 * 0.7 *(p[j + 1, i] - p[j, i]) / (ap_v[j + 1, i] + ap_v[j, i])
    # Fe
    for j in range(0, ny, 1):
        for i in range(0, nx - 1, 1):
            Fe[j, i] = (u[j, i] + u[j, i + 1]) / 2 + 0.7 *(
                        (p_face[1, j, i + 1] - p_face[1, j, i]) / ap_u[j, i + 1] + (p_face[1, j, i] - p_face[3, j, i]) /
                        ap_u[j, i]) / 2 \
                       - 2 * 0.7 *(p[j, i + 1] - p[j, i]) / (ap_u[j, i] + ap_u[j, i + 1])
    # Fs
    for j in range(1, ny, 1):
        for i in range(0, nx, 1):
            Fs[j, i] = (v[j, i] + v[j - 1, i]) / 2 + 0.7 *(
                        (p_face[2, j, i] - p_face[2, j - 1, i]) / ap_v[j - 1, i] + (p_face[0, j, i] - p_face[2, j, i]) /
                        ap_v[j, i]) / 2 \
                       - 2 * 0.7 *(p[j, i] - p[j - 1, i]) / (ap_v[j - 1, i] + ap_v[j, i])
    # Fw
    for j in range(0, ny, 1):
        for i in range(1, nx, 1):
            Fw[j, i] = (u[j, i] + u[j, i - 1]) / 2 + 0.7 *(
                        (p_face[3, j, i] - p_face[3, j, i - 1]) / ap_u[j, i - 1] + (p_face[1, j, i] - p_face[3, j, i]) /
                        ap_u[j, i]) / 2 \
                       - 2 * 0.7 *(p[j, i] - p[j, i - 1]) / (ap_u[j, i - 1] + ap_u[j, i])
    F_info = F_storage(Fn, Fe, Fs, Fw)
    #return (Fn, Fe, Fs, Fw)
    return (F_info)


def make_ps(p, nx, ny):
    p_n = np.zeros((ny, nx))
    p_e = np.zeros((ny, nx))
    p_s = np.zeros((ny, nx))
    p_w = np.zeros((ny, nx))
    p_face = np.zeros((4, ny, nx))
    # p_n
    for i in range(0, nx, 1):  # nx
        for j in range(0, ny - 1, 1):  # ny
            p_face[0, j, i] = (p[j + 1, i] + p[j, i]) / 2
        p_face[0, ny - 1, i] = p[ny - 1, i]
        #p_face[0, ny - 1, i] = 2 * p_face[0, ny - 2, i] - p_face[0, ny - 3, i]
    # p_e
    for j in range(0, ny, 1):  # ny
        for i in range(0, nx - 1, 1):  # nx
            p_face[1, j, i] = (p[j, i] + p[j, i + 1]) / 2
        p_face[1, j, nx - 1] = p[j, nx-1]
        #p_face[1, j, nx - 1] = 2 * p_face[1, j, nx - 2] - p_face[1, j, nx - 3]
    # p_s
    for i in range(0, nx, 1):  # nx
        for j in range(1, ny, 1):  # ny
            p_face[2, j, i] = (p[j, i] + p[j - 1, i]) / 2
        p_face[2, 0, i] = p[0, i]
        p_face[2, 0, i] = 2 * p_face[2, 1, i] - p_face[2, 2, i]
    # p_#w
    for j in range(0, ny, 1):  # ny
        for i in range(1, nx, 1):  # nx
            p_face[3, j, i] = (p[j, i - 1] + p[j, i]) / 2
        p_face[3, j, 0] = p[j,0]
        #p_face[3, j, 0] = 2 * p_face[3, j, 1] - p_face[3, j, 2]
    return (p_face)


def make_xy_cell_centre(nx, ny, dx, dy):
    x_cor = np.zeros((ny, nx))
    y_cor = np.zeros((ny, nx))

    x_cor[:, 0] = dx[0, 0] / 2
    y_cor[0, :] = dy[0, 0] / 2

    for i in range(1, nx, 1):
        x_cor[:, i] = x_cor[:, i - 1] + dx[0, i - 1] / 2 + dx[0, i] / 2
    for j in range(1, ny, 1):
        y_cor[j, :] = y_cor[j - 1, :] + dy[j - 1, 0] / 2 + dy[j, 0] / 2
    return (x_cor, y_cor)

