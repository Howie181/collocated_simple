import time
start_time = time.time()

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
make_u=u_field()
print('Finished loading functions')

# Meshing
nx=120
ny=nx
length=1

dx, dy = mesher.uniform(length, nx, ny)
mesh_info=mesh_storage(nx, ny, dx, dy)
n=nx*ny

# Define BC
Re = 1000
gam = 1/Re
rho = 1
u_n = 1;    u_e = 0;    u_s = 0;    u_w = 0
v_n = 0;    v_e = 0;    v_s = 0;    v_w = 0

bctype_1 = 'fixedValue';    bctype_2 = 'zeroGradient'

north = bctype_1;   east  = bctype_1;   south = bctype_1;   west  = bctype_1

u_bc_info=orientation_bc_storage(north, east, south, west, u_n, u_e, u_s, u_w)
v_bc_info=orientation_bc_storage(north, east, south, west, v_n, v_e, v_s, v_w)

# Step 0) Initialization
ux, uy = 0, 0
x_cor,y_cor=make_xy_cell_centre(nx,ny,dx,dy)
u, v=make_u.uniform(ny, nx, uy, ux)
p=np.zeros((ny,nx))
p_face = make_ps(p, nx, ny)
D, ap_u, ap_v, F_info=initialization_temp(mesh_info, u_bc_info, v_bc_info, u, v, gam, p)

# Step 1) calcualte convection terms fs for all faces
F_info = make_Fs(u, v, nx, ny, p, p_face, ap_u, ap_v)

n = 3000
c_u = 0.01
c_p = 0.001
i = 0
p_prime_avg=np.zeros(n)
p_prime_max=np.zeros(n)
print('Pe:', (dx[0,0]/gam))

while i < n:
    print('Simple itartion no:', i + 1)
    # Step 2) Solve momentum equationsI
    u_star, ap_u = solve_u(mesh_info, u_bc_info, D, F_info, c_u, u, v, p, p_face)
    v_star, ap_v = solve_v(mesh_info, v_bc_info, D, F_info, c_u, u, v, p, p_face)
    step_x=int(nx/3)
    # for j in range(0, step_x, 1):
    #     for k in range(0, step_x, 1):
    #         u_star[j,k]=0
    #         v_star[j,k]=0

    # Step 3) Recalculate convection terms fs
    F_info = make_Fs(u_star, v_star, nx, ny, p, p_face, ap_u, ap_v)

    # Step 4) Solve pressrue correction P'
    p_prime= solve_p(mesh_info, D, F_info, c_p, u, v, p, p_face, ap_u, ap_v)

    # Step 5) Correct u, v, p using P'
    alpha_p = 0.3
    alpha_u = 1
    p, p_prime_2d = p_cor(p, p_prime, ny, nx, alpha_p)

    # for j in range(0, step_x, 1):
    #     for k in range(0, step_x, 1):
    #         p_prime_2d[j,k]=0

    p_prime_face = make_ps(p_prime_2d, nx, ny)
    u, v = u_cor(u_star, v_star, p_prime_face, ap_u, ap_v, ny, nx, alpha_u)

    # Step 6) Correct convection term fs
    F_info = make_Fs(u, v, nx, ny, p, p_face, ap_u, ap_v)

    # Step 7) Update pressure faces
    p_face = make_ps(p, nx, ny)
    p_prime_avg[i]=sum(abs(p_prime_2d[:,:]))/(ny*nx)
    p_prime_max[i]=np.max(abs(p_prime_2d[:,:]))
    no=i
    print('Residual: ',p_prime_max[i])
    if p_prime_max[i]<=1e-4:
        break
    i += 1

print("---Converged in %s Iterations and took %s ---" % (no, time.time() - start_time))

# u_save=np.save("results/step_n=80_u.npy", u)
# v_save=np.save("results/step_n=80_v.npy", v)
# p_save=np.save("results/step_n=80_p.npy", p)

# contour=pp.contour_plot(x_cor, y_cor, p)
contour=pp.contour_plot(x_cor, y_cor, p_prime_2d)
# contour=pp.contour_plot(x_cor, y_cor, u)
# contour=pp.contour_plot(x_cor, y_cor, v)
mag_u = np.zeros((ny,nx))
mag_u[:,:]=sqrt(u**2 + v**2)
contour=pp.streamlines(x_cor, y_cor, u, v, mag_u, Re)

# n_plot=linspace(1,n,n)
# plt.plot(n_plot, p_prime_max, 'm-', marker='x')
# plt.xlim(0, 1000)
# plt.yscale("log")
# plt.show()



print('Average local p\' :', p_prime_avg[no])
print('Maximum local p\' :', p_prime_max[no])