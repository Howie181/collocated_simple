class mesh_storage:
    def __init__(self, nx, ny, dx, dy):
        self.a = nx
        self.b = ny
        self.c = dx
        self.d = dy

class F_storage:
    def __init__(self, Fn, Fe, Fs, Fw):
        self.a = Fn
        self.b = Fe
        self.c = Fs
        self.d = Fw

class orientation_bc_storage:
    def __init__(self, north, east, south, west, phi_n, phi_e, phi_s, phi_w):
        self.a = north
        self.b = east
        self.c = south
        self.d = west
        self.e = phi_n
        self.f = phi_e
        self.g = phi_s
        self.h = phi_w


class face_flux_storage:
    def __init__(self, a_n, a_e, a_s, a_w, a_p):
        self.a = a_n
        self.b = a_e
        self.c = a_s
        self.d = a_w
        self.e = a_p