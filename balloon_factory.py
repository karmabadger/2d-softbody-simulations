import taichi as ti

"""
balloon factor functions
this is for generating grids of particles and springs.
you can generate a few blocks and then add on top of it to make different shapes.
"""

"""
this function creates the grid of particles. That's it
"""


# @ti.pyfunc
# def balloon_maker(n_vertices=3, height=1000):
#     particles_grid = []
#
#     # simple softbody structure
#     # total_size = rows * cols
#
#     for i in range(rows):
#         row_list = []
#         for j in range(cols):
#             particle = p.Blue(2, m=mass, r=radius, pinned=is_pinned, height=height)
#             row_list.append(particle)
#         particles_grid.append(row_list)
#
#     return particles_grid


"""
this function generates all the springs for the grid as to fill it up.
 _
|X|
 -
"""


@ti.pyfunc
def springs_gen(spring_vec, p_len):
    for i in range(p_len):
        # the 2 particles of each springs
        # only set the index
        spring_vec[i][0] = i
        spring_vec[i][1] = (i + 1) % p_len


"""
this sets the position of the particles in the grid all together
start_x, start_y is the middle element 
each particle will be at exactly radius distance from the middle
the first starts at theta = 0

we turn by 2pi/n
"""


@ti.pyfunc
def set_positions(particles_pos, p_len, start_x, start_y, radius):
    dtheta = 2 * 3.1415926 / p_len

    for i in range(p_len):
        x = radius * ti.cos(i * dtheta) + start_x
        y = radius * ti.sin(i * dtheta) + start_y

        particles_pos[i].x = x
        particles_pos[i].y = y


# COULDNT USE SO DIRECTLY COPIED TO ballon.py
@ti.pyfunc
def apply_pressure_forces( p_len, particles_position, particles_forces, pressure_value):
    for i in ti.static(range(p_len)):
        # Get distance vector for the edge
        distance_vector = particles_position[(i) % p_len] - particles_position[(i + 1) % p_len]
        length = distance_vector.norm_sqr()
        # Get normalised normal vector to that edge
        #distance_vector_normal = get_normal(distance_vector) / length
        distance_vector_normal = ti.Vector([-distance_vector[1],distance_vector[0]]).normalized()
        # Calculating pressure force
        F = pressure_value * distance_vector_normal / length
        particles_forces[ (i) % num_particles] += F
        particles_forces[(i+1) % num_particles] += F

# COULDNT USE SO DIRECTLY COPIED TO ballon.py
@ti.pyfunc
def get_area_boundingbox(p_len, particles_position ) -> ti.i32:

    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    for i in range(p_len):
        min_x = min(min_x, particles_position[i].x)
        max_x = max(max_x, particles_position[i].x)
        min_y = min(min_y, particles_position[i].y)
        max_y = max(max_y, particles_position[i].y)

    return ( max_x - min_x )*( max_y - min_y )


@ti.pyfunc
def check_bounding_box(particles_pos, p_len, position):
    is_in = 1.0

    bound = ti.Vector([0.0, 0.0, 0.0, 0.0])
    bound[0] = particles_pos[0].x
    bound[1] = particles_pos[0].x
    bound[2] = particles_pos[0].y
    bound[3] = particles_pos[0].y

    for j in range(1):
        for i in range(p_len):
            vertex = self.vertices[i]
            bound[0] = min(bound[0], vertex[0])
            bound[1] = max(bound[1], vertex[0])
            bound[2] = min(bound[2], vertex[1])
            bound[3] = max(bound[3], vertex[1])

        if position.x < bound[0] or position.x > bound[1] or position.y < bound[2] or position.y > bound[3]:
            is_in = 0.0

    return is_in


@ti.pyfunc
def get_normal(edge):
    normal = ti.Vector([edge[1], -edge[0]])
    return normal



@ti.pyfunc
def draw_shape(gui, particles_pos, p_len, color_shape, thickness):
    for i in range(p_len):
        gui.line(particles_pos[i], particles_pos[(i + 1) % p_len],
    color = color_shape,
    radius = thickness)
