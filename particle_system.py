

import taichi as ti


ti.init(arch=ti.cuda)


@ti.data_oriented
class particle_system:
    particles = []
    number_particles = 0;

    springs = []


    @ti.pyfunc
    def clear_particles(self):

        self.particles = []
        self.springs = []

        number_particles = 0


    @ti.

    

    


    
        