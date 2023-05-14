import taichi as ti
import numpy as np
import integrator as integ
import particle as p
import spring as s
import time
from os import system

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

# todo: PLEASE MANUALLY ADD ALL PARTICLES THAT YOU WISH TO HAVE IN YOUR ENVIRONMENT.
# We will have a 2 linked pendulum. P2 is fixed at the top. P1 connected to P2. P3 connected to P1
p1 = p.Blue(2, m=2, r=5.0, pinned=False)
p2 = p.Blue(2, m=1000000, r=5.0, pinned=True)
p3 = p.Blue(2, m=2, r=5.0, pinned=False)

# todo: PLEASE MANUALLY ADD ALL THE SPRINGS THAT WILL LINK YOUR PARTICLES TOGETHER INTO A SOFTBODY
spring1 = s.Spring(dim=2, r=2.5, k=10000,c=100, p1=p1, p2=p2)  # For a more robust spring, set k to a high value
spring2 = s.Spring(dim=2, r=2.5, k=10000,c=100, p1=p1, p2=p3)  # For a more robust spring, set k to a high value

# todo: PLEASE MANUALLY ADD EACH PARTICLES AND SPRINGS IN THE LISTS BELOW
# Set the particles and springs list
particles = [p1,p2,p3]
springs = [spring1,spring2]

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

# After all taichi objects have been defined, we start initialise the values

# --------------- You can start accessing taichi fields as from this point -------------- #

# TODO: PLEASE SET THE POSITION FOR EACH PARTICLE
# Set position of the particles
p1.set_position(0.4, 0.4)
p2.set_position(0.5, 0.5)
p3.set_position(0.5, 0.3)


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
def redraw_all(gui):
    for i in ti.static(range(len(particles))):
        particles[i].draw(gui)
    for i in ti.static(range(len(springs))):
        springs[i].draw(gui)

cls = lambda: system('cls')

@ti.pyfunc
def main():

    gui = ti.GUI("test", res=(width, height))

    h = step_size/num_substeps

    # For this many time steps
    for i in range(1000000):

        # Resolve particles moving out of the frame
        check_walls_for_all()
        # redraw everything
        redraw_all(gui)

        # Peforms num_substeps steps of size h using RK4 integration
        for k in ti.static(range(num_substeps)):
            integrator.RK4_step(h)

        # update the frame
        gui.show()

        # TODO: If you wish to observe the fields of a spring and particle, do so below
        # Printing values to the terminal for debugging purposes
        print("Count: {} ".format(i))
        spring1.display_values()
        p1.display_values()

        # Uncomment so as to clear the terminal at each time step if you want
        #cls()

if __name__ == "__main__":
    main()
