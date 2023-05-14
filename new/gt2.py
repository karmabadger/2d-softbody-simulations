import taichi as ti
import numpy as np


import integrator as integ
# import Polygon as poly
import grid_factory as gf


import time
from os import system

ti.init(arch=ti.cuda)


"""
all initialization must be done at the very top
"""

# TODO: PLEASE TWEAK THE PARAMETERS BELOW APPROPRIATELY


#drawing debug mode on/off
spring_debug = False
#thickness of the shape
thickness = 1
#color of the shape
color_shape = 0xFF00FF


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




'''
Phase Space is a 2D vector.
1st component contains position.x, position.y, velocity.x, velocity.y
for p1
2nd component contains position.x, position.y, velocity.x, velocity.y
for p2
'''
phase_space = ti.Vector.field(DIM*2, dtype=ti.f32, shape=(len(particles),))

'''
This will store the derivative of the phase space
'''
deriv_phase_space = ti.Vector.field(DIM*2, dtype=ti.f32, shape=(len(particles),))

#This is the integrator used for the simulation
integrator = integ.Integrator(dim=DIM,
                              particles=particles,
                              springs=springs,
                              phase_space=phase_space,
                              deriv_phase_space=deriv_phase_space,
                              useGravity=useGravity,
                              gravity=gravity,
                              viscousDamping=viscousDamping)


# define the polygon
box_vertices = [
    [0, 0.5],
    [1, 0.5],
    [1, 0.6],
    [0, 0.6]
]
# box = poly.Polygon(2, p_vertices=box_vertices, color=0xFF0000)

# After all taichi objects have been defined, we start initialise the values

# --------------- You can start accessing taichi fields as from this point -------------- #

# TODO: PLEASE SET THE POSITION FOR EACH PARTICLE


# Set position of the particles
gf.set_positions(particles_grid, g_rows, g_cols, 0.5, 0.7, 0.02, 0.02)



# Set the rest length to each spring according to the particle position
for spring in springs:
    spring.recomputeRestLength()

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
ground = 0.6
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
    elif particle.position[0].y - particle.r_radius <= ground:
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
def draw_shape(gui, particles):
    size_outer = ti.static(len(particles))
    for i in ti.static(range(size_outer)):
        gui.line(particles[i].position[0], particles[(i+1)%size_outer].position[0], color=color_shape,
                     radius=thickness)

@ti.pyfunc
def draw_shape_from_vertices(gui, vertices, color):
    size_outer = ti.static(len(vertices))
    for i in ti.static(range(size_outer)):
        gui.line(vertices[i], vertices[(i+1)%size_outer], color=color,
                     radius=thickness)


@ti.pyfunc
def redraw_all(gui):

    if spring_debug:
        for i in ti.static(range(len(particles))):
            particles[i].draw(gui)
        for i in ti.static(range(len(springs))):
            springs[i].draw(gui)
    else:
        draw_shape(gui, outer_particles)





cls = lambda: system('cls')

@ti.pyfunc
def main():

    gui = ti.GUI("grid_factory", res=(width, height))

    h = step_size/num_substeps

    # For this many time steps
    for i in range(1000000):

        # Resolve particles moving out of the frame
        check_walls_for_all()
        # redraw everything
        redraw_all(gui)
        # box.draw(gui)

        draw_shape_from_vertices(gui, box_vertices, 0xFF0000)

        # Peforms num_substeps steps of size h using RK4 integration
        for k in ti.static(range(num_substeps)):
            integrator.RK4_step(h)

        # update the frame
        gui.show()

        # TODO: If you wish to observe the fields of a spring and particle, do so below
        # Printing values to the terminal for debugging purposes
        # print("Count: {} ".format(i))
        # spring1.display_values()
        # p1.display_values()

        # Uncomment so as to clear the terminal at each time step if you want
        #cls()

if __name__ == "__main__":
    main()
