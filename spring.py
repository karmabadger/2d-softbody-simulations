import taichi as ti

# import particle as p

# ti.init(arch=ti.cuda)


@ti.data_oriented
class Spring:
    """The Spring"""

    number_of_springs = 0

    def __init__(self, dim, r=1, k=1, c=1, p1=None, p2=None):
        self.dim = dim
        self.p1 = p1  # 1st endpoint of spring
        self.p2 = p2  # 2nd endpoint of spring
        self.k = k  # Spring Stiffness Constant
        self.c = c  # Spring Damping Constant

        self.sid = Spring.number_of_springs

        Spring.number_of_springs += 1
        '''
        if p1 != None and p2 != None: # Rest Length of spring
            self.l0 = (self.p2.position[0] - self.p1.position[0]).norm_sqr()
        else:
            self.l0 = 0
        '''
        self.l0 =  ti.field(float, shape=()) # Rest Length of spring
        self.radius = r  # Radius of spring
        self.distance_temp = ti.Vector.field(self.dim, dtype=ti.f32, shape=(1,))  # Temp placeholder for calculation
        self.color = 0x9999FF


    # @ti.kernel
    # def recomputeRestLength(self):
    #     self.distance_temp[0] = self.p2.position[0] - self.p1.position[0]
    #     self.l0[None] = self.distance_temp[0].norm_sqr()

    @ti.kernel
    def recomputeRestLength(self):
        self.distance_temp[0].x = self.p2.position[0].x - self.p1.position[0].x
        self.distance_temp[0].y = self.p2.position[0].y - self.p1.position[0].y
        self.l0[None] = self.distance_temp[0].norm_sqr()

    '''
    This function returns the current length of the spring
    '''
    @ti.pyfunc
    def get_length(self) -> ti.f32:
        difference_dist_vector = ti.Vector([self.p2.position[0].x - self.p1.position[0].x, self.p2.position[0].y - self.p1.position[0].y])
        return difference_dist_vector.norm_sqr()
    
    '''
    This function applies the spring force on both particles that
    connect the spring
    '''
    @ti.kernel
    def apply_force(self):
        difference_dist_vector = (self.p2.position[0] - self.p1.position[0])
        difference_dist = difference_dist_vector.norm_sqr()
        difference_dist_vector_norm = difference_dist_vector.normalized()

        # Add the spring force due to extension
        self.p1.force[0] += self.k * (difference_dist - self.l0[None]) * difference_dist_vector_norm
        self.p2.force[0] -= self.k * (difference_dist - self.l0[None]) * difference_dist_vector_norm

        difference_vel_vector = (self.p2.velocity[0] - self.p1.velocity[0])

        # Add the spring damping force
        self.p1.force[0] += self.c * (difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm
        self.p2.force[0] -= self.c * (difference_dist_vector_norm.dot(difference_vel_vector)) * difference_dist_vector_norm


    @ti.pyfunc
    def draw(self, gui):
        gui.line(self.p1.position[0], self.p2.position[0], radius=self.radius, color=self.color)

    '''
    Debugging purposes
    '''
    @ti.pyfunc
    def display_values(self):
        print("Rest Length: {} ".format(self.l0[None]))
        print("Current Length: {} ".format(self.get_length()))
        print("Extension: {} ".format(self.get_length() - self.l0[None]))