from numpy import *
from scipy import linalg, sparse

class meshing:
    def uniform(self, length, nx, ny):  # make uniform mesh
        dx=zeros((ny,nx))
        dy=zeros((ny,nx))
        dx[:]=length/nx
        dy[:]=length/ny
        return dx, dy

class u_field:
    def uniform(self, ny, nx, uy, ux):
        u = zeros((ny, nx))
        v = zeros((ny, nx))
        u[:, :] = ux
        v[:, :] = uy
        return(u, v)
    def vortex(self, ny, nx, y_cor, x_cor, length):
        u = zeros((ny, nx))
        v = zeros((ny, nx))
        r = zeros((ny, nx))
        theta = zeros((ny, nx))
        for j in range(0, ny, 1):
            for i in range(0, nx, 1):
                r[j, i] = sqrt((x_cor[j, i] - length / 2) ** 2 + (y_cor[j, i] - length / 2) ** 2)
                theta[j, i] = arctan2((y_cor[j, i] - length / 2), (x_cor[j, i] - length / 2))
                u[j, i] = -r[j, i] * sin(theta[j, i])
                v[j, i] = r[j, i] * cos(theta[j, i])
        return (u, v)
