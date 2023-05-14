import taichi as ti
import numpy as np

import grid_factory as gf

ti.init(arch=ti.cuda)

"""
all initialization must be done at the very top
"""

# TODO: PLEASE TWEAK THE PARAMETERS BELOW APPROPRIATELY
width = 1000
height = 1000
# Dimension - keep it to 2 for now
DIM = 2
# Setting the gravity, viscousDamping to influence the particle motion
gravity = 5
viscousDamping = 3
# Set to true if we wish to have gravity
useGravity = True
# Number of substeps ( range from A1: [ 1 , 100 ] )
num_substeps = 1
# Step size ( range from A1: [ 1e-5 , 1 ] )
step_size = 0.005
# Substep size
h = step_size/num_substeps



# grid system
g_rows = 2
g_cols = 2


# Vector fields related to the particles
particle_radius = 5.0
particle_mass = 2
num_particles = g_rows * g_cols
particle_color = 0x9999FF
x = ti.Vector.field(DIM, dtype=ti.f32, shape=num_particles) # Positions of the particles
v = ti.Vector.field(DIM, dtype=ti.f32, shape=num_particles) # Velocity of the particles
f = ti.Vector.field(DIM, dtype=ti.f32, shape=num_particles) # Forces of the particles

# Vector fields related to the springs
spring_radius = 2.5
spring_stiffness_constant = 10000
spring_damping_constant = 100
num_springs = gf.number_of_springs_estimator(g_rows, g_cols)
spring_color = 0x9999FF
s_connections = ti.Vector.field(2, dtype=ti.i32, shape=num_springs) # Indices of the 2 particles that each spring links
s_rest_lengths = ti.Vector.field(1, dtype=ti.f32, shape=num_springs) # Rest length of each spring

# RK4 Placeholders for computations
phase_space = ti.Vector.field(DIM*2, dtype=ti.f32, shape=num_particles)
k1 = ti.Vector.field(DIM*2, dtype=ti.f32, shape=num_particles)
k2 = ti.Vector.field(DIM*2, dtype=ti.f32, shape=num_particles)
k3 = ti.Vector.field(DIM*2, dtype=ti.f32, shape=num_particles)
k4 = ti.Vector.field(DIM*2, dtype=ti.f32, shape=num_particles)

@ti.pyfunc
def apply_spring_force(index):
    # Get the position of each endpoint of the spring at that index
    pos1 = x[ s_connections[index][0] ]
    pos2 = x[s_connections[index][1]]
    difference_dist_vector = (pos1 - pos2)
    difference_dist = difference_dist_vector.norm_sqr()
    difference_dist_vector_norm = difference_dist_vector.normalized()
    # Get the rest length
    rest_length = s_rest_lengths[index][0]
    # Add the spring force due to extension to each particle connected
    f[s_connections[index][0]] += spring_stiffness_constant * (difference_dist - rest_length) * difference_dist_vector_norm
    f[s_connections[index][1]] -= spring_stiffness_constant * (difference_dist - rest_length) * difference_dist_vector_norm
    # Get the velocity of each endpoint
    vel1 = v[s_connections[index][0]]
    vel2 = v[s_connections[index][1]]
    difference_vel_vector = (vel1 - vel2)
    # Add the spring damping force to each particle connected
    f[s_connections[index][0]] += spring_damping_constant * (difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm
    f[s_connections[index][0]] -= spring_damping_constant * (difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm

@ti.pyfunc
def update_phase_space(phase_space_p):
    # Increment position of each particle manually
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            phase_space_p[i][j] = x[i][j]
            phase_space_p[i][j + DIM] = v[i][j]

@ti.pyfunc
def set_phase_space(phase_space_p):
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            x[i][j] = phase_space_p[i][j]
            v[i][j] = phase_space_p[i][j + DIM]

@ti.pyfunc
def set_deriv_phase_space(phase_space_p, deriv_phase_space_p):
    set_phase_space(phase_space_p)
    # Apply gravity and damping force to each particle
    for i in ti.static(range(num_particles)):
        # Set forces to 0
        for j in ti.static(range(DIM)):
            f[i][j] = 0
        # Add effect of gravity if needed
        if useGravity:
            f[i].y = - gravity * particle_mass
        # Add viscous damping to each dimension
        for j in ti.static(range(DIM)):
            f[i][j] -= viscousDamping * v[i][j]

    # Apply spring force from each spring
    for i in ti.static(range(num_springs)):
        apply_spring_force(i)

    # Set the derivative of the phase space
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            deriv_phase_space_p[i][j] = v[i][j]
            deriv_phase_space_p[i][j+DIM] = f[i][j] / particle_mass

@ti.kernel
def forward_euler_update():

    set_deriv_phase_space(phase_space, k1)

    for i in ti.static(range(num_particles)):
        #if not particle.pinned:
        for j in ti.static(range(DIM*2)):
            phase_space[i][j] += h * k1[i][j]

'''
@ti.pyfunc
def RK4_update():
    set_deriv_phase_space(phase_space,k1)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM*2)):
            self.k2[i][j] = self.phase_space[i][j] + 0.5 * h * self.k1[i][j]

    set_deriv_phase_space(k2, k2)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM * 2)):
            self.k3[i][j] = self.phase_space[i][j] + 0.5 * h * self.k2[i][j]

    set_deriv_phase_space(k3, k3)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM * 2)):
            self.k4[i][j] = self.phase_space[i][j] + h * self.k3[i][j]

    set_deriv_phase_space(k4, k4)

    for i in ti.static(range(num_particles)):
        if self.particles[i].pinned:
            continue
        for j in ti.static(range(self.dim*2)):
            self.phase_space[i][j] += (1 / 6.0) * h * self.k1[i][j]
            self.phase_space[i][j] += (1 / 3.0) * h * self.k2[i][j]
            self.phase_space[i][j] += (1 / 3.0) * h * self.k3[i][j]
            self.phase_space[i][j] += (1 / 6.0) * h * self.k4[i][j]
'''




'''
Function to resolve a particle moving out of the frame if need be
'''
ground = 0.2
@ti.pyfunc
def check_walls(particle_pos, particle_vel):
    if particle_pos.x >= 1:
        if particle_vel.x > 0:
            particle_vel.x = -particle_vel.x
    elif particle_pos.x <= 0:
        if particle_vel.x < 0:
            particle_vel.x = -particle_vel.x
    elif particle_pos.y >= 1:
        if particle_vel.y > 0:
            particle_vel.y = -particle_vel.y
    elif particle_pos.y - (particle_radius / height) <= ground:

        if particle_vel.y < 0:
            particle_vel.y = -particle_vel.y

'''
Function to resolve all particles moving out of frame.
'''
@ti.pyfunc
def check_walls_for_all():
    for i in ti.static(range(num_particles)):
        check_walls(x[i], v[i])


@ti.pyfunc
def draw_ground(gui):
    gui.line([0, ground], [1, ground], radius=1, color=0xFF0000)




@ti.kernel
def step():
    # Increment position of each particle manually
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            x[i][j] += 0.1

@ti.pyfunc
def drawSprings():
    for s in range(num_springs):
        index1 = s_connections[s][0]
        index2 = s_connections[s][1]
        gui.line(x[index1], x[index2], radius=spring_radius, color=spring_color)


@ti.kernel
def initialize():
    # Set position of each particle manually

    gf.set_positions(x, g_rows, g_cols, 0.5, 0.5, 0.02, 0.02)

    # for i in range(num_particles):
    #     print("p: ", x[i])

    # for i in range(num_particles):
    #     print("pind: ", gf.row_col_to_ind(g_rows, g_cols, i))


    # Set velocities and force to 0
    for i in range(num_particles):
        v[i] = ti.Matrix([0, 0])
        f[i] = ti.Matrix([0, 0])

    # Set endpoints of each spring manually
    # s_connections[0] = [0, 1] # connects p0 and p1
    # s_connections[1] = [1, 2] # connects p1 and p2
    # s_connections[2] = [2, 3] # connects p2 and p3
    # s_connections[3] = [3, 0] # connects p3 and p0

    # gf.springs_gen(s_connections, g_rows, g_cols)

    for i in range(num_springs):
        print("s: ", s_connections[i][0], ", ", s_connections[i][1])

    # Set the rest length of each spring
    for i in range(num_springs):
        index1 = s_connections[i][0]
        index2 = s_connections[i][1]
        s_rest_lengths[i][0] = ( x[index1] - x[index2] ).norm_sqr()

# -------------------------- Main -------------------------- #
gui = ti.GUI("test", res=(width, height))
initialize()
# For this many time steps
for i in range(1000000):
    # Drawing the springs and particles on the screen
    check_walls_for_all()
    drawSprings()

    for i in range(num_particles):
        gui.circle(x[i], radius=particle_radius, color=particle_color)
    # gui.circles(x.to_numpy(), radius=particle_radius)
    draw_ground(gui)
    gui.show()
    # Peform 1 Forward Euler step
    for k in ti.static(range(num_substeps)):
        update_phase_space(phase_space)
        forward_euler_update()
        set_phase_space(phase_space)


# ------------------------------------------------------------------------ #
