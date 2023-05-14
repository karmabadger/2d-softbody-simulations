import taichi as ti

import particle as p

# ti.init(arch=ti.cuda)


@ti.data_oriented
class Integrator:
    """The Spring"""

    def __init__(self, dim, particles, springs, phase_space, deriv_phase_space, useGravity, gravity, viscousDamping):
        self.dim = dim
        self.particles = particles
        self.springs = springs
        self.phase_space = phase_space
        self.deriv_phase_space = deriv_phase_space
        self.useGravity = useGravity
        self.gravity = gravity
        self.viscousDamping = viscousDamping
        '''
        Placeholder for computation in the RK4 function
        '''
        self.k1 = ti.Vector.field(dim*2, dtype=ti.f32, shape=(len(particles),))
        self.k2 = ti.Vector.field(dim*2, dtype=ti.f32, shape=(len(particles),))
        self.k3 = ti.Vector.field(dim*2, dtype=ti.f32, shape=(len(particles),))
        self.k4 = ti.Vector.field(dim*2, dtype=ti.f32, shape=(len(particles),))

    '''
    Similar to A1. This is used at each step to
    set the particles positions and velocities to the phase space
    '''
    @ti.pyfunc
    def set_phase_space(self, phase_space = None):
        if not phase_space:
            phase_space = self.phase_space
        for i in ti.static(range(len(self.particles))):
            for j in ti.static(range(self.dim)):
                self.particles[i].position[0][j] = phase_space[i][j]
                self.particles[i].velocity[0][j] = phase_space[i][j+self.dim]

    '''
    Similar to A1. This is used at each step to
    re-update the phase space with the particles position and velocity
    '''
    @ti.pyfunc
    def update_phase_space(self, phase_space = None):
        if not phase_space:
            phase_space = self.phase_space
        for i in ti.static(range(len(self.particles))):
            for j in ti.static(range(self.dim)):
                phase_space[i][j] = self.particles[i].position[0][j]
                phase_space[i][j+self.dim] = self.particles[i].velocity[0][j]

    '''
    This function is called at each time step to 
    compute the derivative of the phase space 
    '''
    @ti.pyfunc
    def set_deriv_phase_space(self, phase_space, deriv_phase_space):
        if not phase_space:
            phase_space = self.phase_space
        if not deriv_phase_space:
            deriv_phase_space = self.deriv_phase_space

        self.set_phase_space(phase_space)
        # Apply gravity and damping force to each particle
        for i in ti.static(range(len(self.particles))):
            particle = self.particles[i]
            particle.clear_force()
            # Add effect of gravity if needed
            if self.useGravity:
                particle.force[0].y = - self.gravity * particle.mass
            # Add viscous damping to each dimension
            for j in ti.static(range(self.dim)):
                particle.force[0][j] -= self.viscousDamping * particle.velocity[0][j]

            #todo: APPLY COLLISION FORCE HERE FOR EACH PARTICLE

        # Apply spring force from each spring
        for i in ti.static(range(len(self.springs))):
            self.springs[i].apply_force()

        # Set the derivative of the phase space
        for i in ti.static(range(len(self.particles))):
            '''
            derivPhaseSpace[i][0] = particles[i].velocity[0].x
            derivPhaseSpace[i][1] = particles[i].velocity[0].y
            derivPhaseSpace[i][2] = particles[i].force[0].x / particles[i].mass
            derivPhaseSpace[i][3] = particles[i].force[0].y / particles[i].mass
            '''
            particle = self.particles[i]
            for j in ti.static(range(self.dim)):
                deriv_phase_space[i][j] = particle.velocity[0][j]
                deriv_phase_space[i][j+self.dim] = particle.force[0][j] / particle.mass

    '''
    This function is called at each time step to update the
    position and velocity of each particle using forward euler
    integration.
    '''
    @ti.pyfunc
    def forward_euler_update(self, h: ti.f32):
        for i in ti.static(range(len(particles))):
            particle = self.particles[i]
            if not particle.pinned:
                for j in ti.static(range(self.dim*2)):
                    self.phase_space[i][j] += h * self.deriv_phase_space[i][j]

    '''
    This function is called at each time step to update the
    position and velocity of each particle using RK4
    integration.
    '''
    @ti.pyfunc
    def RK4_update(self, h: ti.f32):

        self.set_deriv_phase_space(self.phase_space,self.k1)

        for i in ti.static(range(len(self.particles))):
            for j in ti.static(range(self.dim*2)):
                self.k2[i][j] = self.phase_space[i][j] + 0.5 * h * self.k1[i][j]

        self.set_deriv_phase_space(self.k2, self.k2)

        for i in ti.static(range(len(self.particles))):
            for j in ti.static(range(self.dim*2)):
                self.k3[i][j] = self.phase_space[i][j] + 0.5 * h * self.k2[i][j]

        self.set_deriv_phase_space(self.k3, self.k3)

        for i in ti.static(range(len(self.particles))):
            for j in ti.static(range(self.dim*2)):
                self.k4[i][j] = self.phase_space[i][j] + h * self.k3[i][j]

        self.set_deriv_phase_space(self.k4, self.k4)

        for i in ti.static(range(len(self.particles))):
            if self.particles[i].pinned:
                continue
            for j in ti.static(range(self.dim*2)):
                self.phase_space[i][j] += (1 / 6.0) * h * self.k1[i][j]
                self.phase_space[i][j] += (1 / 3.0) * h * self.k2[i][j]
                self.phase_space[i][j] += (1 / 3.0) * h * self.k3[i][j]
                self.phase_space[i][j] += (1 / 6.0) * h * self.k4[i][j]

    '''
    This function is called at each time step to update the
    position and velocity of each particle using RK4
    integration.
    '''
    def RK4_step(self, h: ti.f32):
        '''
        Perform 1 RK4 integration step
        '''
        # Set phase space according to particles posiiton and velocity
        # on last time step
        self.update_phase_space()
        # Update the phase_space according to 1 RK4 step
        self.RK4_update(h)
        # Set the particles position and velocity to the phase space
        self.set_phase_space()