import taichi as ti
import numpy as np
import math

import itertools

ti.init(arch=ti.cuda)

"""
all initialization must be done at the very top
"""

spring_debug = False

write_image = True
image_dir = "img/square_pinned/"
project_name = "square_pinned"

# TODO: PLEASE TWEAK THE PARAMETERS BELOW APPROPRIATELY
width = 1000
height = 1000
GROUND = 0.3
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
step_size = 0.01
# Substep size
h = step_size / num_substeps

# Vector fields related to the particles
particle_radius = 5.0
particle_mass = 2
num_particles = 4
particle_color = 0x9999FF
x = np.zeros((num_particles,DIM),dtype=float)  # Positions of the particles
v = np.zeros((num_particles,DIM),dtype=float)  # Velocity of the particles
f = np.zeros((num_particles,DIM),dtype=float)  # Forces of the particles
pinned = np.zeros(num_particles,dtype=int)  # Pinned status of particles

# soft body object shape parameters
color_shape = 0xFF00FF
thickness = 3

# Vector fields related to the springs
spring_radius = 1
spring_stiffness_constant = 100000
spring_damping_constant = 100
num_springs = 6
spring_color = 0x9999FF
s_connections = np.zeros((num_springs,2), dtype=int)  # Indices of the 2 particles that each spring links
s_rest_lengths = np.zeros(num_springs, dtype=float)  # Rest length of each spring

# RK4 Placeholders for computations
phase_space = np.zeros((num_particles,DIM * 2), dtype=float)
k1 = np.zeros((num_particles,DIM * 2), dtype=float)
k2 = np.zeros((num_particles,DIM * 2), dtype=float)
k3 = np.zeros((num_particles,DIM * 2), dtype=float)
k4 = np.zeros((num_particles,DIM * 2), dtype=float)


def apply_spring_force(index):
    # Get the position of each endpoint of the spring at that index
    pos1 = x[s_connections[index][0]]
    pos2 = x[s_connections[index][1]]
    dist_v = (pos2 - pos1)
    dist_sqrt = dist_v.dot(dist_v)
    difference_dist_vector_norm = dist_v / math.sqrt(dist_sqrt)
    # Get the rest length
    rest_length = s_rest_lengths[index]
    # Add the spring force due to extension to each particle connected
    f[s_connections[index][0]] += spring_stiffness_constant * (
                dist_sqrt - rest_length) * difference_dist_vector_norm
    f[s_connections[index][1]] -= spring_stiffness_constant * (
                dist_sqrt - rest_length) * difference_dist_vector_norm
    # Get the velocity of each endpoint
    vel1 = v[s_connections[index][0]]
    vel2 = v[s_connections[index][1]]
    difference_vel_vector = (vel2 - vel1)
    # Add the spring damping force to each particle connected
    f[s_connections[index][0]] += spring_damping_constant * (
        difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm
    f[s_connections[index][0]] -= spring_damping_constant * (
        difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm


def update_phase_space(phase_space_p):
    # Increment position of each particle manually
    for i in range(num_particles):
        for j in range(DIM):
            phase_space_p[i][j] = x[i][j]
            phase_space_p[i][j + DIM] = v[i][j]



def set_phase_space(phase_space_p):
    for i in range(num_particles):
        # If not pinned
        if pinned[i] != 1:
            for j in range(DIM):
                x[i][j] = phase_space_p[i][j]
                v[i][j] = phase_space_p[i][j + DIM]



def set_deriv_phase_space(phase_space_p, deriv_phase_space_p):
    set_phase_space(phase_space_p)
    # Apply gravity and damping force to each particle
    for i in range(num_particles):
        # Set forces to 0
        for j in range(DIM):
            f[i][j] = 0
        # Add effect of gravity if needed
        if useGravity:
            f[i][1] = - gravity * particle_mass
        # Add viscous damping to each dimension
        for j in range(DIM):
            f[i][j] -= viscousDamping * v[i][j]

    # Apply spring force from each spring
    for i in range(num_springs):
        apply_spring_force(i)

    # Set the derivative of the phase space
    for i in range(num_particles):
        for j in range(DIM):
            deriv_phase_space_p[i][j] = v[i][j]
            deriv_phase_space_p[i][j + DIM] = f[i][j] / particle_mass



def forward_euler_update():
    set_deriv_phase_space(phase_space, k1)

    for i in range(num_particles):
        for j in range(DIM * 2):
            phase_space[i][j] += h * k1[i][j]



def RK4_update():
    # print("rk4.0")
    # ti.sync()
    # ti.async_flush()

    set_deriv_phase_space(phase_space, k1)

    for i in range(num_particles):
        for j in range(DIM * 2):
            k2[i][j] = phase_space[i][j] + 0.5 * h * k1[i][j]

    # print("rk4.1")
    # ti.sync()
    # ti.async_flush()
    set_deriv_phase_space(k2, k2)

    for i in range(num_particles):
        for j in range(DIM * 2):
            k3[i][j] = phase_space[i][j] + 0.5 * h * k2[i][j]

    # print("rk4.2")
    # ti.sync()
    # ti.async_flush()
    set_deriv_phase_space(k3, k3)

    for i in range(num_particles):
        for j in range(DIM * 2):
            k4[i][j] = phase_space[i][j] + h * k3[i][j]

    # print("rk4.3")
    # ti.sync()
    # ti.async_flush()
    set_deriv_phase_space(k4, k4)

    for i in range(num_particles):
        for j in range(DIM * 2):
            phase_space[i][j] += (1 / 6.0) * h * k1[i][j]
            phase_space[i][j] += (1 / 3.0) * h * k2[i][j]
            phase_space[i][j] += (1 / 3.0) * h * k3[i][j]
            phase_space[i][j] += (1 / 6.0) * h * k4[i][j]

    # print("rk4.4")
    # ti.sync()
    # ti.async_flush()

def forward_euler_step():
    # print("loopb ", i, ", ", k)
    update_phase_space(phase_space)
    # print("loopb.1 ", i, ", ", k)
    forward_euler_update()
    # print("loopb.2 ", i, ", ", k)
    set_phase_space(phase_space)

def rk4_step():
    # print("loopb ", i, ", ", k)
    update_phase_space(phase_space)
    # print("loopb.1 ", i, ", ", k)
    RK4_update()

    # print("loopb.2 ", i, ", ", k)
    set_phase_space(phase_space)

def check_ground():
    for i in range(num_particles):
        if x[i][0] >= 1:
            if v[i][0] > 0:
                v[i][0] = -v[i][0]
        elif x[i][0] <= 0:
            if v[i][0] < 0:
                v[i][0] = -v[i][0]
        elif x[i][1] >= 1:
            if v[i][1] > 0:
                v[i][1] = -v[i][1]
        elif x[i][1] - (particle_radius / height) <= GROUND:
            if v[i][1] < 0:
                v[i][1] = -v[i][1]



def drawSprings():
    for s in range(num_springs):
        index1 = s_connections[s][0]
        index2 = s_connections[s][1]
        gui.line(x[index1], x[index2], radius=spring_radius, color=spring_color)


def initialize():


    # todo: SET YOUR PARTICLES POSITION HERE
    # Set position of each particle manually
    x[0] = [0.5,0.7]
    x[1] = [0.4,0.6]
    x[2] = [0.5,0.5]
    x[3] = [0.6,0.6]

    # todo: SET PINNED STATUS HERE
    # Set pinned status for particle if need be
    pinned[1] = 1 # The 3rd particle is pinned now

    # todo: SET SPRING CONNECTIONS HERE
    # Set endpoints of each spring manually
    s_connections[0] = [0, 1] # connects p0 and p1
    s_connections[1] = [1, 2] # connects p1 and p2
    s_connections[2] = [2, 3] # connects p2 and p3
    s_connections[3] = [3, 0] # connects p3 and p0
    s_connections[4] = [1, 3]  # connects p3 and p0
    s_connections[5] = [0, 2]  # connects p3 and p0

    # Set the rest length of each spring
    for i in range(num_springs):
        index1 = s_connections[i][0]
        index2 = s_connections[i][1]
        tmp = x[index1] - x[index2]
        s_rest_lengths[i] = tmp.dot(tmp)


# -------------------------- Main -------------------------- #
gui = ti.GUI("test", res=(width, height))
initialize()


colors = np.array([0x0033FF, 0xFF3333], dtype=np.uint32)
# For this many time steps
for i in range(10000):
    # Drawing the ground, springs and particles on the screen
    gui.line([0, GROUND], [1, GROUND], radius=2, color=0x617763)

    gui.circles(x, radius=particle_radius, color=colors[pinned])
    drawSprings()

    if ti.static(write_image):
        ti.imwrite(gui.get_image(),
                   image_dir + project_name + "_" + str(i) + ".png")

    gui.show()
    # Peform 1 RK4 step
    for k in ti.static(range(num_substeps)):
        rk4_step()

    # Draws the ground
    check_ground()

# ------------------------------------------------------------------------ #
