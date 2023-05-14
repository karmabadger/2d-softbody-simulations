import taichi as ti

class Particle:
    number_particles = 0
    def __init__(self, dim, m=1.0, r=1.0, pinned = False, height=1000):

        self.pid = Particle.number_particles
        Particle.number_particles += 1
        self.dim = dim
        self.mass = m
        self.radius = r
        self.r_radius = r/height

        self.position = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        self.velocity = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        self.force = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        self.distance = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        self.pinned = pinned
        if pinned:
            self.color = 0xFF0000
        else:
            self.color = 0x9999FF

    @ti.pyfunc
    def set_position(self, new_x: ti.float32, new_y: ti.float32):
        self.position[0].x = new_x
        self.position[0].y = new_y

    @ti.pyfunc
    def set_position3D(self, new_x: ti.float32, new_y: ti.float32, new_z: ti.float32):
        self.position[0].x = new_x
        self.position[0].y = new_y
        self.position[0].z = new_z

    @ti.pyfunc
    def apply_force(self, dt: ti.f32):
        acc = self.force[0] * (1 / self.mass)
        self.velocity[0] += acc * dt

    # '''
    # Simple incremenetal step. Not really used
    # '''
    #
    # @ti.kernel
    # def move(self, dt: ti.f32):
    #     if not self.pinned:
    #         diff = self.velocity[0] * dt
    #         self.position[0] += diff

    # @ti.kernel
    # def apply_force(self):

    @ti.pyfunc
    def set_force(self, fx: ti.f32, fy: ti.f32):
        self.force[0].x = fx
        self.force[0].y = fy

    # @ti.kernel
    # def set_force3D(self, fx: ti.f32, fy: ti.f32, fz: ti.f32):
    #     self.force[0].x = fx
    #     self.force[0].y = fy
    #     self.force[0].z = fz

    @ti.pyfunc
    def clear_force(self):
        for i in ti.static(range(self.dim)):
            self.force[0][i] = 0

    @ti.pyfunc
    def set_velocity(self, fx: ti.f32, fy: ti.f32):
        self.velocity[0].x = fx
        self.velocity[0].y = fy

    # @ti.kernel
    # def set_velocity3D(self, fx: ti.f32, fy: ti.f32, fz: ti.f32):
    #     self.velocity[0].x = fx
    #     self.velocity[0].y = fy
    #     self.velocity[0].z = fz

    @ti.pyfunc
    def add_force(self, fx: ti.f32, fy: ti.f32):
        self.force[0].x += fx
        self.force[0].y += fy

    # @ti.kernel
    # def add_force3D(self, fx: ti.f32, fy: ti.f32, fz: ti.f32):
    #     self.force[0].x += fx
    #     self.force[0].y += fy
    #     self.force[0].z += fz

    '''
       This function returns the magnitude of the force acting on that particle
    '''

    @ti.pyfunc
    def get_mag_force(self) -> ti.f32:
        return self.force[0].norm_sqr()

    @ti.pyfunc
    def draw(self, gui):
        gui.circle(self.position[0], radius=self.radius, color=self.color)

    '''
    Debugging purposes
    '''

    @ti.pyfunc
    def display_values(self):
        print("F_x : {} ".format(self.force[0].x))
        print("F_y : {} ".format(self.force[0].y))
        if self.dim == 3:
            print("F_z : {} ".format(self.force[0].z))
        print("F : {} ".format(self.get_mag_force()))