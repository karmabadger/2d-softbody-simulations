import taichi as ti

# import particle as p
# import spring as s


"""
grid factor functions
this is for generating grids of particles and springs.
you can generate a few blocks and then add on top of it to make different shapes.
"""

# """
# this function creates the grid of particles. That's it
# """
# @ti.pyfunc
# def grid_maker(rows=1, cols=1, mass=2, radius=5.0, is_pinned = False, height=1000):
#     particles_grid = []
#
#     # simple softbody structure
#     # total_size = rows * cols
#
#     for i in range(rows):
#         row_list = []
#         for j in range(cols):
#             particle = p.Particle(2, m=mass, r=radius, pinned=is_pinned, height=height)
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
def springs_gen(spring_connections, rows, cols, p_len):
    # all horizontal springs
    spring_index = 0
    for u in range(1):
        for i in range(rows):
            for j in range(cols - 1):
                spring_connections[spring_index][0] = row_col_to_ind(rows, cols, i, j)
                spring_connections[spring_index][1] = row_col_to_ind(rows, cols, i, j + 1)
                # print(i, ",", j , "s in 1: ", spring_connections[spring_index][0], ", ", spring_connections[spring_index][1])

                spring_index += 1

    # for i in range(p_len):
    #     print("s in total1: ", spring_connections[i][0], ", ", spring_connections[i][1])

    # all vertical springs
    for u in range(1):
        for i in range(rows - 1):
            for j in range(cols):
                spring_connections[spring_index][0] = row_col_to_ind(rows, cols, i, j)
                spring_connections[spring_index][1] = row_col_to_ind(rows, cols, i + 1, j)
                # print(i, ",", j , "s in 2: ", spring_connections[spring_index][0], ", ", spring_connections[spring_index][1])
                spring_index += 1

    # for i in range(p_len):
    #     print("s in total2: ", spring_connections[i][0], ", ", spring_connections[i][1])
    # diagonal springs

    # \ springs
    for u in range(1):
        for i in range(rows - 1):
            for j in range(cols - 1):
                spring_connections[spring_index][0] = row_col_to_ind(rows, cols, i, j)
                spring_connections[spring_index][1] = row_col_to_ind(rows, cols, i + 1, j + 1)
                # print("s in 3: ", spring_connections[spring_index][0], ", ", spring_connections[spring_index][1])
                spring_index += 1
                # spring_item = s.Spring(dim=dim, r=radius, k=k, c=c, p1=p_grid[i][j], p2=p_grid[i + 1][j + 1])
                # springs_list.append(spring_item)

    # for i in range(p_len):
    #     print("s in total3: ", spring_connections[i][0], ", ", spring_connections[i][1])

    # / springs
    for u in range(1):
        for i in range(rows - 1):
            for j in range(1, (cols)):
                spring_connections[spring_index][0] = row_col_to_ind(rows, cols, i, j)
                spring_connections[spring_index][1] = row_col_to_ind(rows, cols, i + 1, j - 1)
                # print("s in 4: ", spring_connections[spring_index][0], ", ", spring_connections[spring_index][1])
                spring_index += 1
                # spring_item = s.Spring(dim=dim, r=radius, k=k, c=c, p1=p_grid[i][j], p2=p_grid[i + 1][j - 1])
                # springs_list.append(spring_item)

    # for i in range(p_len):
    #     print("s in total4: ", spring_connections[i][0], ", ", spring_connections[i][1])


"""
this function gets the outer layer of the grid into a list.
this is for drawing the shape.
"""


@ti.pyfunc
def get_outer_particles(rows, cols):
    outer_particles = []

    # # all the elements of the first row
    # row = p_grid[0]

    for i in range(cols):
        outer_particles.append(row_col_to_ind(rows, cols, 0, i))

    # all the elements of the last column
    for i in range(1, (rows - 1)):
        outer_particles.append(row_col_to_ind(rows, cols, i, cols - 1))

    # all the elements of the last row
    for i in range(cols - 1, -1, -1):
        outer_particles.append(row_col_to_ind(rows, cols, rows - 1, i))

    # all the elements of the first column
    for i in range((rows - 2), 0, -1):
        outer_particles.append(row_col_to_ind(rows, cols, i, 0))

    return outer_particles


# """
# this turns the grid multi dimensional array into a list
# """
# @ti.pyfunc
# def grid_to_list(p_grid, rows, cols):
#     particles = []
#
#     for i in range(rows):
#         for j in ti.static(range(cols)):
#             particles.append(p_grid[i][j])
#
#     return particles


"""
this sets the position of the particles in the grid all together
the [0][0] element starts at start_x, start_y and for each we add dx or dy
"""


@ti.pyfunc
def set_positions(particles_pos, rows, cols, start_x, start_y, dx, dy):
    for i in range(rows):
        for j in range(cols):
            x = start_x + j * dx
            y = start_y + i * dy
            # print("r: ", rows, ", c: ", cols, " i: ", i, " j: ", j, " index: ", cols * i + j, " x: ", x, ",", y)
            particles_pos[cols * i + j][0] = x
            particles_pos[cols * i + j][1] = y


@ti.pyfunc
def total_size(rows, cols):
    return rows * cols


@ti.pyfunc
def number_of_springs_estimator(rows, cols):
    return (rows - 1) * cols + (cols - 1) * rows + (rows - 1) * (cols - 1) * 2


def row_col_to_ind(rows, cols, r_ind, c_ind):
    index = cols * r_ind + c_ind
    return index


# @ti.pyfunc
# def draw_grid(gui, rows, cols, particle_pos, use_debug_spring):
#     if use_debug_spring:
#         for i in ti.static(range(len(particles))):
#             particles[i].draw(gui)
#         for i in ti.static(range(len(springs))):
#             springs[i].draw(gui)
#     else:
#         for i in ti.static(range(len(particles))):
#             draw_shape(gui)
#
@ti.pyfunc
def draw_shape(gui, particles_pos, outer_particles_indices, out_len, color_shape, thickness):
    for i in range(out_len):
        gui.line(particles_pos[outer_particles_indices[i]], particles_pos[outer_particles_indices[(i + 1) % out_len]],
    color = color_shape,
    radius = thickness)
