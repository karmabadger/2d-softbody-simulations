import taichi as ti
import numpy as np

import grid_factory as gf

ti.init(arch=ti.cuda, debug=True, log_level=ti.TRACE)

"""
all initialization must be done at the very top
"""

# TODO: PLEASE TWEAK THE PARAMETERS BELOW APPROPRIATELY

spring_debug = False

write_image = True
image_dir = "img/grid/"
project_name = "grid"

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
step_size = 0.005
# Substep size
h = step_size / num_substeps

# grid system
g_rows = 10
g_cols = 10

# Vector fields related to the particles
particle_radius = 5.0
particle_mass = 2
num_particles = g_rows * g_cols
particle_color = 0x9999FF
x = ti.Vector.field(DIM, dtype=ti.f32, shape=num_particles)  # Positions of the particles
v = ti.Vector.field(DIM, dtype=ti.f32, shape=num_particles)  # Velocity of the particles
f = ti.Vector.field(DIM, dtype=ti.f32, shape=num_particles)  # Forces of the particles
pinned = ti.field(dtype=int, shape=num_particles)  # Pinned status of particles

# soft body object shape parameters
# outer_particles_indices = []

# out_length = 0
color_shape = 0xFF00FF
thickness = 1

# Vector fields related to the springs
spring_radius = 2.5
spring_stiffness_constant = 1000000
spring_damping_constant = 100
num_springs = gf.number_of_springs_estimator(g_rows, g_cols)
spring_color = 0x9999FF
s_connections = ti.Vector.field(2, dtype=ti.i32, shape=num_springs)  # Indices of the 2 particles that each spring links
s_rest_lengths = ti.Vector.field(1, dtype=ti.f32, shape=num_springs)  # Rest length of each spring

# RK4 Placeholders for computations
phase_space = ti.Vector.field(DIM * 2, dtype=ti.f32, shape=num_particles)
k1 = ti.Vector.field(DIM * 2, dtype=ti.f32, shape=num_particles)
k2 = ti.Vector.field(DIM * 2, dtype=ti.f32, shape=num_particles)
k3 = ti.Vector.field(DIM * 2, dtype=ti.f32, shape=num_particles)
k4 = ti.Vector.field(DIM * 2, dtype=ti.f32, shape=num_particles)


# @ti.func
# def apply_spring_force(index):
#     # Get the position of each endpoint of the spring at that index
#     pos1 = x[s_connections[index][0]]
#     pos2 = x[s_connections[index][1]]
#     difference_dist_vector = (pos2 - pos1)
#     difference_dist = difference_dist_vector.norm_sqr()
#     difference_dist_vector_norm = difference_dist_vector.normalized()
#     # Get the rest length
#     rest_length = s_rest_lengths[index][0]
#     # Add the spring force due to extension to each particle connected
#     f[s_connections[index][0]] += spring_stiffness_constant * (
#             difference_dist - rest_length) * difference_dist_vector_norm
#     f[s_connections[index][1]] -= spring_stiffness_constant * (
#             difference_dist - rest_length) * difference_dist_vector_norm
#     # Get the velocity of each endpoint
#     vel1 = v[s_connections[index][0]]
#     vel2 = v[s_connections[index][1]]
#     difference_vel_vector = (vel2 - vel1)
#     # Add the spring damping force to each particle connected
#     f[s_connections[index][0]] += spring_damping_constant * (
#         difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm
#     f[s_connections[index][0]] -= spring_damping_constant * (
#         difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm


# @ti.func
# def update_phase_space(phase_space_p):
#     # Increment position of each particle manually
#
#     for u in range(1):
#         for i in ti.static(range(num_particles)):
#             for j in ti.static(range(DIM)):
#                 phase_space_p[i][j] = x[i][j]
#                 phase_space_p[i][j + DIM] = v[i][j]


# @ti.func
# def set_phase_space(phase_space_p):
#     for u in range(1):
#         for i in ti.static(range(num_particles)):
#             # If not pinned
#             if pinned[i] != 1:
#                 for j in ti.static(range(DIM)):
#                     x[i][j] = phase_space_p[i][j]
#                     v[i][j] = phase_space_p[i][j + DIM]

# @ti.kernel
def detect_circle_collision(p1, p2):
    tmp = np.array([p1.position[0][0] - p2.position[0][0],
                     p1.position[0][1] - p2.position[0][1]])
    # dist_norm = tmp.norm(0.0000001)
    dist_norm = np.linalg.norm(tmp)

    if dist_norm <= ((p1.radius + p2.radius) / height):
        # detected collision
        tangent1 = np.array([-tmp[1], tmp[0]])

        tangent1_mag = np.linalg.norm(tangent1)

        unit_tangent = tangent1 / tangent1_mag

        rel_vel = np.array([p1.velocity[0][0] - p2.velocity[0][0], p1.velocity[0][1] - p2.velocity[0][1]])

        length = rel_vel.dot(unit_tangent)

        vel_comp_on_tangent = length * unit_tangent
        vel_perp_tangent = rel_vel - vel_comp_on_tangent

        p1.velocity[0][0] -= vel_perp_tangent[0]
        p1.velocity[0][1] -= vel_perp_tangent[1]

        p2.velocity[0][0] += vel_perp_tangent[0]
        p2.velocity[0][1] += vel_perp_tangent[1]

        return


@ti.func
def set_deriv_phase_space(phase_space_p, deriv_phase_space_p):
    # set_phase_space(phase_space_p)

    for i in ti.static(range(num_particles)):
        # If not pinned
        if pinned[i] != 1:
            for j in ti.static(range(DIM)):
                x[i][j] = phase_space_p[i][j]
                v[i][j] = phase_space_p[i][j + DIM]


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
        # apply_spring_force(i)
        index = i
        # Get the position of each endpoint of the spring at that index
        pos1 = x[s_connections[index][0]]
        pos2 = x[s_connections[index][1]]
        difference_dist_vector = (pos2 - pos1)
        difference_dist = difference_dist_vector.norm_sqr()
        difference_dist_vector_norm = difference_dist_vector.normalized()
        # Get the rest length
        rest_length = s_rest_lengths[index][0]


        # Add the spring force due to extension to each particle connected
        force = spring_stiffness_constant * (
                difference_dist - rest_length) * difference_dist_vector_norm
        f[s_connections[index][0]] += force
        f[s_connections[index][1]] -= force
        # Get the velocity of each endpoint
        vel1 = v[s_connections[index][0]]
        vel2 = v[s_connections[index][1]]
        difference_vel_vector = (vel2 - vel1)


        # Add the spring damping force to each particle connected
        damp_force = spring_damping_constant * (
            difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm
        f[s_connections[index][0]] += damp_force
        f[s_connections[index][0]] -= damp_force

    # Set the derivative of the phase space
    for i in ti.static(range(num_particles)):
        for j in ti.static(range(DIM)):
            deriv_phase_space_p[i][j] = v[i][j]
            deriv_phase_space_p[i][j + DIM] = f[i][j] / particle_mass


# @ti.kernel
# def forward_euler_update():
#     set_deriv_phase_space(phase_space, k1)
#
#     for i in ti.static(range(num_particles)):
#         for j in ti.static(range(DIM * 2)):
#             phase_space[i][j] += h * k1[i][j]


# @ti.func
# def RK4_update():
#     set_deriv_phase_space(phase_space, k1)
#
#     for i in ti.static(range(num_particles)):
#         for j in ti.static(range(DIM * 2)):
#             k2[i][j] = phase_space[i][j] + 0.5 * h * k1[i][j]
#
#     set_deriv_phase_space(k2, k2)
#
#     for i in ti.static(range(num_particles)):
#         for j in ti.static(range(DIM * 2)):
#             k3[i][j] = phase_space[i][j] + 0.5 * h * k2[i][j]
#
#     set_deriv_phase_space(k3, k3)
#
#     for i in ti.static(range(num_particles)):
#         for j in ti.static(range(DIM * 2)):
#             k4[i][j] = phase_space[i][j] + h * k3[i][j]
#
#     set_deriv_phase_space(k4, k4)
#
#     for i in ti.static(range(num_particles)):
#         for j in ti.static(range(DIM * 2)):
#             phase_space[i][j] += (1 / 6.0) * h * k1[i][j]
#             phase_space[i][j] += (1 / 3.0) * h * k2[i][j]
#             phase_space[i][j] += (1 / 3.0) * h * k3[i][j]
#             phase_space[i][j] += (1 / 6.0) * h * k4[i][j]


@ti.kernel
def rk4_step():
    for u in range(1):
        for k in ti.static(range(num_substeps)):
            # update_phase_space(phase_space)


            for i in ti.static(range(num_particles)):
                for j in ti.static(range(DIM)):
                    phase_space[i][j] = x[i][j]
                    phase_space[i][j + DIM] = v[i][j]


            # RK4_update()
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



            # set_phase_space(phase_space)

            for i in ti.static(range(num_particles)):
                # If not pinned
                if pinned[i] != 1:
                    for j in ti.static(range(DIM)):
                        x[i][j] = phase_space[i][j]
                        v[i][j] = phase_space[i][j + DIM]



# @ti.kernel
# def step():
#     # Increment position of each particle manually
#     for i in ti.static(range(num_particles)):
#         for j in ti.static(range(DIM)):
#             x[i][j] += 0.1


@ti.kernel
def check_ground():
    for i in range(num_particles):
        if x[i].x >= 1:
            if v[i].x > 0:
                v[i].x = -v[i].x
        elif x[i].x <= 0:
            if v[i].x < 0:
                v[i].x = -v[i].x
        elif x[i].y >= 1:
            if v[i].y > 0:
                v[i].y = -v[i].y
        elif x[i].y - (particle_radius / height) <= GROUND:
            if v[i].y < 0:
                v[i].y = -v[i].y


@ti.pyfunc
def drawSprings():
    for s in range(num_springs):
        index1 = s_connections[s][0]
        index2 = s_connections[s][1]
        gui.line(x[index1], x[index2], radius=spring_radius, color=spring_color)


@ti.kernel
def initialize():
    # Set velocities and force to 0 and pinned to False ( 0 )
    for i in range(num_particles):
        # v[i] = ti.Matrix([0, 0])
        v[i][0] = 0
        v[i][1] = 0
        # f[i] = ti.Matrix([0, 0])
        f[i][0] = 0
        f[i][1] = 0
        pinned[i] = 0

    # todo: SET YOUR PARTICLES POSITION HERE
    # Set position of each particle manually
    # x[0] = [0.45,0.45]
    # x[1] = [0.45,0.5]
    # x[2] = [0.5,0.5]
    # x[3] = [0.5,0.45]

    gf.set_positions(x, g_rows, g_cols, 0.5, 0.5, 0.02, 0.02)

    # todo: SET PINNED STATUS HERE
    # Set pinned status for particle if need be
    # pinned[2] = 1 # The 3rd particle is pinned now

    # todo: SET SPRING CONNECTIONS HERE
    # Set endpoints of each spring manually
    # s_connections[0] = [0, 1] # connects p0 and p1
    # s_connections[1] = [1, 2] # connects p1 and p2
    # s_connections[2] = [2, 3] # connects p2 and p3
    # s_connections[3] = [3, 0] # connects p3 and p0

    gf.springs_gen(s_connections, g_rows, g_cols, num_springs)

    # for i in range(num_springs):
    #     print("s: ", s_connections[i][0], ", ", s_connections[i][1])

    # Set the rest length of each spring
    for i in range(num_springs):
        index1 = s_connections[i][0]
        index2 = s_connections[i][1]
        s_rest_lengths[i][0] = (x[index1] - x[index2]).norm_sqr()


# -------------------------- Main -------------------------- #

print("running...")
gui = ti.GUI(project_name, res=(width, height))
print("initializing...")
initialize()
print("getting outer particles...")
outer_particles_indices = gf.get_outer_particles(g_rows, g_cols)
out_length = len(outer_particles_indices)

colors = np.array([0x0033FF, 0xFF3333], dtype=np.uint32)
# For this many time steps
for i in range(1000):
    # print("getting outer particles...")
    # Drawing the ground, springs and particles on the screen
    gui.line([0, GROUND], [1, GROUND], radius=2, color=0x617763)



    if spring_debug:
        drawSprings()
        gui.circles(x.to_numpy(), radius=particle_radius, color=colors[pinned.to_numpy()])

        if write_image:
            ti.imwrite(gui.get_image(),
                   image_dir + project_name + "_debug_" + str(g_rows) + "x" + str(g_cols) + "_" + str(i) + ".png")
    else:

        gf.draw_shape(gui, x, outer_particles_indices, out_length, color_shape, thickness)

        if write_image:
            ti.imwrite(gui.get_image(),
                   image_dir + project_name + "_" + str(g_rows) + "x" + str(g_cols) + "_" + str(i) + ".png")

    gui.show()

    # Peform 1 RK4 step
    rk4_step()
    # Checks the ground
    check_ground()

# ------------------------------------------------------------------------ #
