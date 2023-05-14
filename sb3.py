import taichi as ti
import numpy as np

import particle as p
import spring as s
import Polygon as poly

import grid_factory as gf
import time
from os import system




ti.init(arch=ti.gpu)

"""
all initialization must be done at the very top
"""

# drawing debug mode on/off
spring_debug = False

# thickness of the shape
thickness = 1

# color of the shape
color_shape = 0xFF00FF

# screen ratios
width = 1000
height = 1000



# # simple softbody structure
g_rows = 6
g_cols = 6

total_size = g_rows * g_cols
n_springs = (g_rows - 1) * g_cols + (g_cols - 1) * g_rows + (g_rows - 1) * (g_cols - 1) * 2


particles_grid = gf.grid_maker(g_rows, g_cols)
springs = gf.springs_gen(particles_grid, g_rows, g_cols, 2, 2.5, 500000, 100)
outer_particles = gf.get_outer_particles(particles_grid, g_rows, g_cols)

particles = gf.grid_to_list(particles_grid, g_rows, g_cols)

# particles[0].pinned = True



num_particles = total_size

# Setting the gravity, viscousDamping to influence the particle motion
gravity = 5
viscousDamping = 3
# Set to true if we wish to have gravity
useGravity = True

# num_substeps range from A1: [ 1 , 100 ]
num_substeps = 1
# Steps_size range from A1: [ 1e-5 , 1 ]
step_size = 0.005

# placeholder vector field for computations
tmp = ti.Vector.field(2, dtype=ti.f32, shape=(num_particles,))

'''
Phase Space is a 2D vector.
1st component contains position.x, position.y, velocity.x, velocity.y
for p1
2nd component contains position.x, position.y, velocity.x, velocity.y
for p2
'''
phase_space = ti.Vector.field(4, dtype=ti.f32, shape=(num_particles,))

'''
This will store the derivative of the phase space
'''
deriv_phase_space = ti.Vector.field(4, dtype=ti.f32, shape=(num_particles,))

'''
Placeholder for computation in the RK4 function
'''
k1 = ti.Vector.field(4, dtype=ti.f32, shape=(num_particles,))
k2 = ti.Vector.field(4, dtype=ti.f32, shape=(num_particles,))
k3 = ti.Vector.field(4, dtype=ti.f32, shape=(num_particles,))
k4 = ti.Vector.field(4, dtype=ti.f32, shape=(num_particles,))









# define the polygon
box_vertices = [
    [0, 0.5],
    [1, 0.5],
    [1, 0.6],
    [0, 0.6]
]
# box = poly.Polygon(2,  p_vertices=box_vertices, color=0xFF0000)

# After all taichi objects have been defined, we start initialise the values

# Set position of the particles

gf.set_positions(particles_grid, g_rows, g_cols, 0.5, 0.7, 0.02, 0.02)
# Set the rest length according to the particle position

for spring in springs:
    spring.recomputeRestLength()



'''
Similar to A1. This is used at each step to
set the particles positions and velocities to the phase space
'''


@ti.pyfunc
def set_phase_space(phaseSpace):
    for i in ti.static(range(len(particles))):
        particles[i].position[0].x = phaseSpace[i][0]
        particles[i].position[0].y = phaseSpace[i][1]
        particles[i].velocity[0].x = phaseSpace[i][2]
        particles[i].velocity[0].y = phaseSpace[i][3]


'''
Similar to A1. This is used at each step to
re-update the phase space with the particles position and velocity
'''


@ti.pyfunc
def update_phase_space(phaseSpace):
    for i in ti.static(range(len(particles))):
        phaseSpace[i][0] = particles[i].position[0].x
        phaseSpace[i][1] = particles[i].position[0].y
        phaseSpace[i][2] = particles[i].velocity[0].x
        phaseSpace[i][3] = particles[i].velocity[0].y


'''
This function is called at each time step to 
compute the derivative of the phase space 
'''


@ti.pyfunc
def set_deriv_phase_space(phaseSpace, derivPhaseSpace):
    set_phase_space(phaseSpace)
    # Apply gravity and damping force to each particle
    for i in ti.static(range(len(particles))):
        particle = particles[i]
        particle.clear_force()
        # Add effect of gravity if needed
        if useGravity:
            particle.force[0].y = - gravity * particle.mass
        particle.force[0].x -= viscousDamping * particle.velocity[0].x
        particle.force[0].y -= viscousDamping * particle.velocity[0].y

    # Apply spring force from each spring
    for i in ti.static(range(len(springs))):
        springs[i].apply_force()

    # Set the derivative of the phase space
    for i in ti.static(range(len(particles))):
        derivPhaseSpace[i][0] = particles[i].velocity[0].x
        derivPhaseSpace[i][1] = particles[i].velocity[0].y
        derivPhaseSpace[i][2] = particles[i].force[0].x / particles[i].mass
        derivPhaseSpace[i][3] = particles[i].force[0].y / particles[i].mass


'''
This function is called at each time step to update the
position and velocity of each particle using forward euler
integration.
'''


@ti.pyfunc
def forward_euler_step(dt, derivPhaseSpace):
    for i in ti.static(range(len(particles))):
        particle = particles[i]
        if not particle.pinned:
            particle.position[0].x += dt * derivPhaseSpace[i][0]
            particle.position[0].y += dt * derivPhaseSpace[i][1]
            particle.velocity[0].x += dt * derivPhaseSpace[i][2]
            particle.velocity[0].y += dt * derivPhaseSpace[i][3]


'''
This function is called at each time step to update the
position and velocity of each particle using RK4
integration.
'''


@ti.pyfunc
def RK4_step(dt, phaseSpace):
    set_deriv_phase_space(phaseSpace, k1)

    for i in ti.static(range(len(particles))):
        for j in ti.static(range(4)):
            k2[i][j] = phaseSpace[i][j] + 0.5 * dt * k1[i][j]

    set_deriv_phase_space(k2, k2)

    for i in ti.static(range(len(particles))):
        for j in ti.static(range(4)):
            k3[i][j] = phaseSpace[i][j] + 0.5 * dt * k2[i][j]

    set_deriv_phase_space(k3, k3)

    for i in ti.static(range(len(particles))):
        for j in ti.static(range(4)):
            k4[i][j] = phaseSpace[i][j] + dt * k3[i][j]

    set_deriv_phase_space(k4, k4)

    for i in ti.static(range(len(particles))):
        if particles[i].pinned:
            continue
        for j in ti.static(range(4)):
            phaseSpace[i][j] += (1 / 6.0) * dt * k1[i][j]
            phaseSpace[i][j] += (1 / 3.0) * dt * k2[i][j]
            phaseSpace[i][j] += (1 / 3.0) * dt * k3[i][j]
            phaseSpace[i][j] += (1 / 6.0) * dt * k4[i][j]


'''
Function to clear the force acting on all particles in this system
'''


@ti.pyfunc
def clear_forces_for_all():
    for i in ti.static(range(len(particles))):
        particles[i].clear_force()


'''
Function to resolve a particle moving out of the frame if need be
'''


@ti.pyfunc
def check_walls(particle):
    if particle.position[0].x >= 1:
        if particle.velocity[0].x > 0:
            particle.velocity[0].x = -particle.velocity[0].x
    elif particle.position[0].x <= 0:
        if particle.velocity[0].x < 0:
            particle.velocity[0].x = -particle.velocity[0].x
    elif particle.position[0].y >= 1:
        if particle.velocity[0].y > 0:
            particle.velocity[0].y = -particle.velocity[0].y
    elif particle.position[0].y - particle.r_radius <= 0.6:
        if particle.velocity[0].y < 0:
            particle.velocity[0].y = -particle.velocity[0].y


'''
Function to resolve all particles moving out of frame.
'''


@ti.pyfunc
def check_walls_for_all():
    for i in ti.static(range(len(particles))):
        check_walls(particles[i])


'''
Function to draw every particles and spring in one frame
'''


@ti.pyfunc
def draw_shape(gui):
    size_outer = ti.static(len(outer_particles))
    for i in ti.static(range(len(outer_particles))):
        gui.line(outer_particles[i].position[0], outer_particles[(i + 1) % size_outer].position[0], color=color_shape,
                 radius=thickness)


@ti.pyfunc
def redraw_all(gui):
    if spring_debug:
        for i in ti.static(range(len(particles))):
            particles[i].draw(gui)
        for i in ti.static(range(len(springs))):
            springs[i].draw(gui)
    else:
        for i in ti.static(range(len(particles))):
            draw_shape(gui)


@ti.pyfunc
def mag(v):
    magnitude = ti.sqrt(v[0].x ** 2 + v[0].y ** 2)
    return magnitude


@ti.pyfunc
def debug(vector, position, gui):
    tmp[0].x = position[0].x + vector[0].x
    tmp[0].y = position[0].y + vector[0].y
    gui.line(position[0], tmp[0], color=0xFFFFFF, radius=1)


cls = lambda: system('cls')


@ti.pyfunc
def main():
    gui = ti.GUI("test", res=(width, height))

    old_t = time.time()
    new_t = time.time()

    h = step_size / num_substeps

    # Printing values to the terminal for debugging purposes
    # print("Initially")
    # spring1.display_values()
    # p1.display_values()

    # For this many time steps
    for i in range(1000000):
        # Get dt
        old_t = new_t
        new_t = time.time()
        dt = new_t - old_t

        # Resolve particles moving out of the frame and
        # redraw everything
        check_walls_for_all()
        redraw_all(gui)
        # box.draw(gui)

        for k in ti.static(range(num_substeps)):
            '''
            Perform 1 RK4 integration step
            '''
            # Set phase space according to particles posiiton and velocity
            # on last time step
            update_phase_space(phase_space)
            # Update the phase_space according to 1 RK4 step
            RK4_step(h, phase_space)
            # Set the particles position and velocity to the state space
            set_phase_space(phase_space)
            # for particle in particles:
            #     box.detect_circle_polygon_collision(particle)

        # update the frame
        gui.show()

        # Printing values to the terminal for debugging purposes
        # print("Count: {} ".format(i))
        # spring1.display_values()
        # p1.display_values()

        # Uncomment so as to clear the terminal at each time step
        # cls()


if __name__ == "__main__":
    main()
