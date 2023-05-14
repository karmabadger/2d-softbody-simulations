import taichi as ti
import numpy as np
import bf2 as bf

import itertools

ti.init(arch=ti.cuda)

"""
all initialization must be done at the very top
"""

# TODO: PLEASE TWEAK THE PARAMETERS BELOW APPROPRIATELY


spring_debug = True

self_collision = True

write_image = False
image_dir = "img/pressure/"
project_name = "pressure"

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
num_substeps = 1000
# Step size ( range from A1: [ 1e-5 , 1 ] )
step_size = 0.002
# Substep size
h = step_size / num_substeps

# # balloon system
balloon_radius = 0.03
start_x = 0.5
start_y = 0.45

# Pressure Constants
# P =  V^-1 * N * R * T
use_pressure_model = True  # set to true to use pressure model
draw_center = True  # Set to true to see center for the triangulation
use_triangulation = True  # Set to true for exact area using triangulation,
#  or False for simple bounding area
R = 8.314  # Keep constant
N = 1000  # Keep Constant
T = 30  # Keep Constant
scaling_factor = 5e-6  # Vary accordingly

# Vector fields related to the particles
particle_radius = 5
particle_mass = 2
num_particles = 8
particle_color = 0x9999FF
x = np.zeros((num_particles, DIM), dtype=float)  # Positions of the particles
v = np.zeros((num_particles, DIM), dtype=float)  # Velocity of the particles
f = np.zeros((num_particles, DIM), dtype=float)  # Forces of the particles
pinned = np.zeros(num_particles, dtype=int)  # Pinned status of particles

# soft body object shape
# the particles are the outer particles. only one layer of particles
color_shape = 0xFF00FF
thickness = 1

# Vector fields related to the springs
spring_radius = 2.5
spring_stiffness_constant = 10000000000
spring_damping_constant = 100
num_springs = num_particles
spring_color = 0x9999FF
s_connections = np.zeros((num_springs, 2), dtype=int)  # Indices of the 2 particles that each spring links
s_rest_lengths = np.zeros(num_springs, dtype=float)  # Rest length of each spring

# RK4 Placeholders for computations
phase_space = np.zeros((num_particles, DIM * 2), dtype=float)
k1 = np.zeros((num_particles, DIM * 2), dtype=float)
k2 = np.zeros((num_particles, DIM * 2), dtype=float)
k3 = np.zeros((num_particles, DIM * 2), dtype=float)
k4 = np.zeros((num_particles, DIM * 2), dtype=float)


# @ti.pyfunc
def apply_spring_force(index):
    # Get the position of each endpoint of the spring at that index
    pos1 = x[s_connections[index][0]]
    pos2 = x[s_connections[index][1]]
    difference_dist_vector = (pos2 - pos1)
    difference_dist = difference_dist_vector.dot(difference_dist_vector)

    difference_dist_vector_mag = np.linalg.norm(difference_dist_vector)
    difference_dist_vector_norm = difference_dist_vector / difference_dist_vector_mag
    # Get the rest length
    rest_length = s_rest_lengths[index]
    # Add the spring force due to extension to each particle connected
    f[s_connections[index][0]] += spring_stiffness_constant * (
            difference_dist - rest_length) * difference_dist_vector_norm
    f[s_connections[index][1]] -= spring_stiffness_constant * (
            difference_dist - rest_length) * difference_dist_vector_norm
    # Get the velocity of each endpoint
    vel1 = v[s_connections[index][0]]
    vel2 = v[s_connections[index][1]]
    difference_vel_vector = (vel2 - vel1)
    # Add the spring damping force to each particle connected
    f[s_connections[index][0]] += spring_damping_constant * (
        difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm
    f[s_connections[index][0]] -= spring_damping_constant * (
        difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm


# @ti.pyfunc
def update_phase_space(phase_space_p):
    # Increment position of each particle manually
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            phase_space_p[i][j] = x[i][j]
            phase_space_p[i][j + DIM] = v[i][j]


# @ti.pyfunc
def set_phase_space(phase_space_p):
    for i in ti.static(range(num_particles)):
        # If not pinned
        if pinned[i] != 1:
            for j in ti.static(range(DIM)):
                x[i][j] = phase_space_p[i][j]
                v[i][j] = phase_space_p[i][j + DIM]


# @ti.pyfunc
def apply_pressure_forces(pressure_value):
    for i in ti.static(range(num_particles)):
        # Get distance vector for the edge
        distance_vector = x[(i) % num_particles] - x[(i + 1) % num_particles]
        length = distance_vector.dot(distance_vector)
        # Get normalised normal vector to that edge
        # distance_vector_normal = get_normal(distance_vector) / length
        # distance_vector_normal = ti.Vector([-distance_vector[1], distance_vector[0]]).normalized()
        distance_vector_normal = np.array([-distance_vector[1], distance_vector[0]])
        distance_vector_normal_norm = np.linalg.norm(distance_vector_normal)
        distance_vector_normal_normalized = distance_vector_normal / distance_vector_normal_norm
        # Calculating pressure force
        F = pressure_value * distance_vector_normal_normalized / length
        f[(i) % num_particles] += F
        f[(i + 1) % num_particles] += F


# @ti.pyfunc
def get_area_boundingbox():
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    for i in ti.static(range(num_particles)):
        min_x = min(min_x, x[i].x)
        max_x = max(max_x, x[i].x)
        min_y = min(min_y, x[i].y)
        max_y = max(max_y, x[i].y)

    return (max_x - min_x) * (max_y - min_y)


# @ti.pyfunc
def get_center():
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    for i in ti.static(range(num_particles)):
        min_x = min(min_x, x[i][0])
        max_x = max(max_x, x[i][0])
        min_y = min(min_y, x[i][1])
        max_y = max(max_y, x[i][1])

    return np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])


# @ti.pyfunc
def get_area_triangulation():
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    for i in ti.static(range(num_particles)):
        min_x = min(min_x, x[i][0])
        max_x = max(max_x, x[i][0])
        min_y = min(min_y, x[i][1])
        max_y = max(max_y, x[i][1])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    area = 0.0

    # accumulate area of each small triangles that make up the balloon
    for i in ti.static(range(num_particles)):
        # Get distance vector for the edge
        V1 = x[(i) % num_particles]
        V2 = x[(i + 1) % num_particles]
        cross = (V1[0] - center_x) * (V2[1] - center_y) - (V1[1] - center_y) * (V2[0] - center_x)
        area += abs(cross) / 2

    return area


# @ti.pyfunc
def set_deriv_phase_space(phase_space_p, deriv_phase_space_p):
    set_phase_space(phase_space_p)
    # Apply gravity and damping force to each particle
    for i in ti.static(range(num_particles)):
        # Set forces to 0
        for j in ti.static(range(DIM)):
            f[i][j] = 0
        # Add effect of gravity if needed
        if useGravity:
            f[i][1] = - gravity * particle_mass
        # Add viscous damping to each dimension
        for j in ti.static(range(DIM)):
            f[i][j] -= viscousDamping * v[i][j]

    # Apply spring force from each spring
    for i in range(num_springs):
        apply_spring_force(i)

    # apply pressure forces
    if use_pressure_model:
        # Couldnt use bf ( some compile time error occurred )
        # area = bf.get_area_triangulation(num_particles,x)
        # pressure_value = scaling_factor * N * R * T / area
        # bf.apply_pressure_forces(num_particles,x,f,pressure_value)
        if use_triangulation:
            area = get_area_triangulation()
            pressure_value = scaling_factor * N * R * T / area
            apply_pressure_forces(pressure_value)
        else:
            area = get_area_boundingbox()
            pressure_value = scaling_factor * N * R * T / area
            apply_pressure_forces(pressure_value)

    # Set the derivative of the phase space
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            deriv_phase_space_p[i][j] = v[i][j]
            deriv_phase_space_p[i][j + DIM] = f[i][j] / particle_mass


# @ti.kernel
def forward_euler_update():
    set_deriv_phase_space(phase_space, k1)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM * 2)):
            phase_space[i][j] += h * k1[i][j]


# @ti.kernel
def RK4_update():
    set_deriv_phase_space(phase_space, k1)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM * 2)):
            k2[i][j] = phase_space[i][j] + 0.5 * h * k1[i][j]

    set_deriv_phase_space(k2, k2)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM * 2)):
            k3[i][j] = phase_space[i][j] + 0.5 * h * k2[i][j]

    set_deriv_phase_space(k3, k3)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM * 2)):
            k4[i][j] = phase_space[i][j] + h * k3[i][j]

    set_deriv_phase_space(k4, k4)

    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM * 2)):
            phase_space[i][j] += (1 / 6.0) * h * k1[i][j]
            phase_space[i][j] += (1 / 3.0) * h * k2[i][j]
            phase_space[i][j] += (1 / 3.0) * h * k3[i][j]
            phase_space[i][j] += (1 / 6.0) * h * k4[i][j]


# @ti.kernel
def step():
    # Increment position of each particle manually
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            x[i][j] += 0.1


# @ti.kernel
def detect_circle_collision(p1, p2):
    tmp = np.array([x[p1][0] - x[p2][0],
                    x[p1][1] - x[p2][1]])
    # dist_norm = tmp.norm(0.0000001)
    dist_norm = np.linalg.norm(tmp)

    if dist_norm <= ((particle_radius * 2) / height):
        # detected collision
        tangent1 = np.array([-tmp[1], tmp[0]])

        tangent1_mag = np.linalg.norm(tangent1)

        unit_tangent = tangent1 / tangent1_mag

        rel_vel = np.array([v[p1][0] - v[p2][0], v[p1][1] - v[p2][1]])

        length = rel_vel.dot(unit_tangent)

        vel_comp_on_tangent = length * unit_tangent
        vel_perp_tangent = rel_vel - vel_comp_on_tangent

        v[p1][0] -= vel_perp_tangent[0]
        v[p1][1] -= vel_perp_tangent[1]

        v[p2][0] += vel_perp_tangent[0]
        v[p2][1] += vel_perp_tangent[1]

        return


# @ti.kernel
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

            diff_y = ((particle_radius / height) + GROUND)-x[i][1]
            x[i][1] = diff_y + (particle_radius / height) + GROUND
            if v[i][1] < 0:
                v[i][1] = -v[i][1]


# @ti.pyfunc
def drawSprings():
    for s in range(num_springs):
        index1 = s_connections[s][0]
        index2 = s_connections[s][1]
        gui.line(x[index1], x[index2], radius=spring_radius, color=spring_color)


# @ti.kernel
def initialize():
    # # Set velocities and force to 0 and pinned to False ( 0 )
    for i in range(num_particles):
        v[i] = np.array([0,-10])
    #     f[i] = ti.Matrix([0, 0])
    #     pinned[i] = 0

    # todo: SET YOUR PARTICLES POSITION HERE
    # Set position of each particle manually
    # x[0] = [0.45,0.45]
    # x[1] = [0.45,0.5]
    # x[2] = [0.5,0.5]
    # x[3] = [0.5,0.45]

    bf.set_positions(x, num_particles, start_x, start_y, balloon_radius)

    # todo: SET PINNED STATUS HERE
    # Set pinned status for particle if need be
    # pinned[num_particles // 4] = 1  #

    # todo: SET SPRING CONNECTIONS HERE
    # Set endpoints of each spring manually
    # s_connections[0] = [0, 1] # connects p0 and p1
    # s_connections[1] = [1, 2] # connects p1 and p2
    # s_connections[2] = [2, 3] # connects p2 and p3
    # s_connections[3] = [3, 0] # connects p3 and p0

    bf.springs_gen(s_connections, num_springs)

    # Set the rest length of each spring
    for i in range(num_springs):
        index1 = s_connections[i][0]
        index2 = s_connections[i][1]
        tmp = x[index1] - x[index2]
        # s_rest_lengths[i][0] = tmp.dot(tmp)
        s_rest_lengths[i] = tmp.dot(tmp)


# -------------------------- Main -------------------------- #
gui = ti.GUI("test", res=(width, height))
initialize()
colors = np.array([0x0033FF, 0xFF3333], dtype=np.uint32)

list_particle_indices = list(range(num_particles))
list_combination = list(itertools.combinations(list_particle_indices, 2))
# For this many time steps
for i in range(1000000):
    if draw_center:
        gui.circle(get_center(), radius=particle_radius, color=0xFFFFFF)
    # Drawing the ground, springs and particles on the screen
    gui.line([0, GROUND], [1, GROUND], radius=2, color=0x617763)

    if spring_debug:
        drawSprings()
        gui.circles(x, radius=particle_radius, color=colors[pinned])

        if write_image:
            ti.imwrite(gui.get_image(),
                       image_dir + project_name + "_debug_" + str(num_particles) + "_" + str(i) + ".png")
    else:

        bf.draw_shape(gui, x, num_particles, color_shape, thickness)

        if write_image:
            ti.imwrite(gui.get_image(),
                       image_dir + project_name + "_" + str(num_particles) + "_" + str(i) + ".png")

    gui.show()
    # Peform 1 RK4 step
    for k in ti.static(range(num_substeps)):
        update_phase_space(phase_space)
        RK4_update()
        set_phase_space(phase_space)
    # Draws the ground
    check_ground()

    if self_collision:
        for i, j in list_combination:
            # print(i)
            detect_circle_collision(i,j)

# ------------------------------------------------------------------------ #