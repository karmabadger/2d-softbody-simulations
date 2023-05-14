import taichi as ti

import particle as p
import spring as s


"""
grid factor functions
this is for generating grids of particles and springs.
you can generate a few blocks and then add on top of it to make different shapes.
"""


"""
this function creates the grid of particles. That's it
"""
@ti.pyfunc
def grid_maker(rows=1, cols=1, mass=2, radius=5.0, is_pinned = False, height=1000):
    particles_grid = []

    # simple softbody structure
    # total_size = rows * cols

    for i in range(rows):
        row_list = []
        for j in range(cols):
            particle = p.Blue(2, m=mass, r=radius, pinned=is_pinned, height=height)
            row_list.append(particle)
        particles_grid.append(row_list)

    return particles_grid


"""
this function generates all the springs for the grid as to fill it up.
 _
|X|
 -
"""
@ti.pyfunc
def springs_gen(p_grid, rows, cols, dim=2, radius=2.5, k=100000, c=100):
    springs_list = []

    # all horizontal springs
    for i in range(rows):
        for j in range(cols - 1):
            spring_item = s.Spring(dim=dim, r=radius, k=k, c=c, p1=p_grid[i][j], p2=p_grid[i][j + 1])
            springs_list.append(spring_item)

    # all vertical springs
    for i in range(rows - 1):
        for j in ti.static(range(cols)):
            spring_item = s.Spring(dim=dim, r=radius, k=k, c=c, p1=p_grid[i][j], p2=p_grid[i + 1][j])
            springs_list.append(spring_item)


    # diagonal springs

    # \ springs
    for i in range(rows - 1):
        for j in ti.static(range(cols - 1)):
            spring_item = s.Spring(dim=dim, r=radius, k=k, c=c, p1=p_grid[i][j], p2=p_grid[i + 1][j + 1])
            springs_list.append(spring_item)


    # rem = len(springs_list)

    # counter = 0
    # / springs
    for i in range(rows - 1):
        for j in range(1, (cols)):
            # counter += 1
            spring_item = s.Spring(dim=dim, r=radius, k=k, c=c, p1=p_grid[i][j], p2=p_grid[i + 1][j - 1])
            springs_list.append(spring_item)

    # print("counter: ", counter, ", real: ", (len(springs_list) - rem))

    return springs_list



"""
this function gets the outer layer of the grid into a list.
this is for drawing the shape.
"""
@ti.pyfunc
def get_outer_particles(p_grid, rows, cols):
    outer_particles = []

    # all the elements of the first row
    row = p_grid[0]

    for i in range(cols):
        if row[i]:
            outer_particles.append(row[i])

    # all the elements of the last column
    for i in range(1, (rows - 1)):
        if p_grid[i][-1]:
            outer_particles.append(p_grid[i][-1])


    # all the elements of the last row
    row = p_grid[-1]

    for i in range(cols - 1, -1, -1):
        if row[i]:
            outer_particles.append(row[i])

    # all the elements of the first column
    for i in range(1, (rows - 1)):
        if p_grid[i][0]:
            outer_particles.append(p_grid[i][0])

    return outer_particles


"""
this turns the grid multi dimensional array into a list
"""
@ti.pyfunc
def grid_to_list(p_grid, rows, cols):
    particles = []

    for i in range(rows):
        for j in ti.static(range(cols)):
            particles.append(p_grid[i][j])

    return particles


"""
this sets the position of the particles in the grid all together
the [0][0] element starts at start_x, start_y and for each we add dx or dy
"""
@ti.pyfunc
def set_positions(p_grid, rows, cols, start_x, start_y, dx, dy):
    for i in range(cols):
        for j in range(rows):
            x = start_x + i * dx
            y = start_y + j * dy
            # print("x: ", x, ", y: ", y)
            cur_p = p_grid[j][i]
            cur_p.set_position(x, y)


@ti.pyfunc
def total_size(rows, cols):
    return rows * cols

@ti.pyfunc
def number_of_springs_estimator(rows, cols):
    return (rows - 1) * cols + (cols - 1) * rows + (rows - 1) * (cols - 1) * 2
