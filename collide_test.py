# fractal.py
import taichi as ti
import numpy as np

import particle as p
import time
import Polygon as poly

# initialize
ti.init(arch=ti.cuda)


# width and height of the screen
width = 1000
height = 1000

#gravity
g = -9.8


# temporary vectors for usage later
tmp = ti.Vector.field(2, dtype=ti.f32, shape=(2,))
tmp2 = ti.Vector.field(2, dtype=ti.f32, shape=(2,))


# list of particles
# should be replaced by a particle system
list_particles = []
list_particles.append(p.Blue(2, m=1000, r=5))
list_particles.append(p.Blue(2, m=1000000, r=5))

list_particles[0].r_radius = 10/height
list_particles[1].r_radius = 10/height

 # list of polygons
p_vertices=[[0, 0.25],[1, 0.25],[1, 0.3], [0, 0.3]]
polygons = []
polygons.append(poly.Polygon(2, pinned=False, mass=1.0, p_vertices=p_vertices, color=0xFFFFFF))

list_particles[0].set_position(0.6, 0.5)
list_particles[1].set_position(0.5, 0.5)


gui = ti.GUI("test", res=(width, height))





# @ti.pyfunc
# def check_walls(particle):
#     if particle.position[0].x + particle.radius >= 1:
#         if particle.velocity[0].x > 0:
#             particle.velocity[0].x = -particle.velocity[0].x
#     elif particle.position[0].x - particle.radius  <= 0:
#         if particle.velocity[0].x < 0:
#             particle.velocity[0].x = -particle.velocity[0].x
#     elif particle.position[0].y + particle.radius >= 1:
#         if particle.velocity[0].y > 0:
#             particle.velocity[0].y = -particle.velocity[0].y
#     elif particle.position[0].y - particle.radius <= 0:
#         if particle.velocity[0].y < 0:
#             particle.velocity[0].y = -particle.velocity[0].y


@ti.pyfunc
def check_walls(particle):
    if particle.position[0].x >= 1:
        if particle.velocity[0].x > 0:
            particle.velocity[0].x = -particle.velocity[0].x
    elif particle.position[0].x  <= 0:
        if particle.velocity[0].x < 0:
            particle.velocity[0].x = -particle.velocity[0].x
    elif particle.position[0].y >= 1:
        if particle.velocity[0].y > 0:
            particle.velocity[0].y = -particle.velocity[0].y
    elif particle.position[0].y  <= 0:
        if particle.velocity[0].y < 0:
            particle.velocity[0].y = -particle.velocity[0].y


@ti.pyfunc
def clear_forces():

    list_particles[0].set_force(0.0, 0.0)
    list_particles[1].set_force(0.0, 0.0)



# @ti.kernel
# def apply_g():
#     # for i in ti.static(range(0, p1.dim)):
#     #     tmp[0][i] = p1.position[0][i] - p2.position[0][i]
#
#     tmp[0].x = list_particles[0].x - p2.position[0].x
#     tmp[0].y = list_particles[0].y - p2.position[0].y
#     r2 = tmp[0].norm_sqr()
#     g = (p.G * ((p1.mass * p2.mass) / r2))
#
#     p1.force[0] -= g * tmp[0].normalized()
#     p2.force[0] += g * tmp[0].normalized()


@ti.pyfunc
def mag(v):
    magnitude = ti.sqrt(v[0].x ** 2 + v[0].y ** 2)
    return magnitude


@ti.pyfunc
def debug(vector, position, gui):
    tmp[0].x = position[0].x + vector[0].x
    tmp[0].y = position[0].y + vector[0].y
    gui.line(position[0], tmp[0], color = 0xFFFFFF, radius = 1)


"""
circle to circle collisions
"""
@ti.kernel
def detect_circle_collision(p1: ti.template(), p2: ti.template()):
    tmp[0].x = list_particles[0].position[0].x - list_particles[1].position[0].x
    tmp[0].y = list_particles[0].position[0].y - list_particles[1].position[0].y
    dist_norm = tmp[0].norm(0.0000001)

    if dist_norm <= ((p1.radius + p2.radius) / height):
        # detected collision
        tangent1 = ti.Vector([-tmp[0].y, tmp[0].x])

        unit_tangent = tangent1.normalized()

        rel_vel = ti.Vector([p1.velocity[0].x - p2.velocity[0].x, p1.velocity[0].y - p2.velocity[0].y])

        length = rel_vel.dot(unit_tangent)

        vel_comp_on_tangent = length * unit_tangent
        vel_perp_tangent = rel_vel - vel_comp_on_tangent

        p1.velocity[0].x -= vel_perp_tangent.x
        p1.velocity[0].y -= vel_perp_tangent.y

        p2.velocity[0].x += vel_perp_tangent.x
        p2.velocity[0].y += vel_perp_tangent.y

        return

    @ti.kernel
    def detect_circle_polygon_collision(polygon: ti.template(), p: ti.template()):

        bbox = polygon.check_bounding_box(p.position)
        # if bbox != 0:

        bbox = polygon.check_same_side(p.position)

        if bbox != 0:
            # collided with bounding box
            #     print("collide i: ", self.i_count)
            #     self.inc_count_i()

            distances_mag = []

            for i in range(polygon.number_vertices):
                polygon.distances[i] = p.position[0] - polygon.vertices[i]
                print("v: ", v.vertices[i], " p: ", p.position[0], " d: ", polygon.distances[i])

                distances_mag.append(polygon.distances[i].norm())

            polygon.position[0] = (polygon.vertices[0] + polygon.distances[0])
            print("p1: ", polygon.position[0])

            # edges = []
            for i in range(polygon.number_vertices):
                polygon.edges[i] = polygon.vertices[(i + 1) % polygon.number_vertices] - polygon.vertices[i]
                # edge = ti.Vector(self.vertices[(i+1)%self.number_vertices].x - self.vertices[i].x, self.vertices[(i+1)%self.number_vertices].y - self.vertices[i].y)
                # edges.append(edge)

            # normals = []
            normals_mag = []
            for i in ti.static(range(polygon.number_vertices)):
                # print("i", i)
                # cur_dist = ti.Vector([distances[i].x, distances[i].y])
                dot = polygon.distances[i].dot(polygon.edges[i])
                dot_vec = dot * polygon.edges[i]
                polygon.normals[i] = (polygon.distances[i] - dot_vec).normalized()

                normals_mag.append(polygon.normals[i].norm_sqr())
            #     normals.append(normal.normalized())
            #
            #
            self.shortest_ind[None] = 0
            shortest_dist = normals_mag[0]
            for unknown in range(1):
                for i in ti.static(range(self.number_vertices)):
                    if shortest_dist < normals_mag[i]:
                        shortest_dist = normals_mag[i]
                        self.shortest_ind[None] = i

            # bounce
            change = p.velocity[0].dot(self.normals[self.shortest_ind])

    # tmp[0].x = p1.position[0].x - p2.position[0].x
    # tmp[0].y = p1.position[0].y - p2.position[0].y
    # dist_norm = tmp[0].norm(0.0000001)
    #
    #
    # if dist_norm <= ((p1.radius + p2.radius) / height):
    #     # detected collision
    #
    #     # tangents
    #     tangent1 = ti.Vector([-tmp[0].y, tmp[0].x])
    #
    #     unit_tangent = tangent1.normalized()
    #
    #     rel_vel = ti.Vector([p1.velocity[0].x - p2.velocity[0].x, p1.velocity[0].y - p2.velocity[0].y])
    #
    #     length = rel_vel.dot(unit_tangent)
    #
    #     vel_comp_on_tangent = length * unit_tangent
    #     vel_perp_tangent = rel_vel - vel_comp_on_tangent
    #
    #     p1.velocity[0].x -= vel_perp_tangent.x
    #     p1.velocity[0].y -= vel_perp_tangent.y
    #
    #     p2.velocity[0].x += vel_perp_tangent.x
    #     p2.velocity[0].y += vel_perp_tangent.y
    #
    #     return

@ti.kernel
def apply_g(p: ti.template()):
    p.force[0] += g

@ti.pyfunc
def main():

    # gui = ti.GUI("test", res=(width, height))

    old_t = time.time()
    new_t = time.time()

    # list_particles[0].set_velocity(0.02, 0.2)
    # list_particles[1].set_velocity(0.05, 0.2)

    for i in ti.static(range(1000000)):
        old_t = new_t
        new_t = time.time()
        dt = new_t - old_t


        clear_forces()

        apply_g(list_particles[0])
        apply_g(list_particles[1])

        list_particles[0].apply_force(dt)
        list_particles[1].apply_force(dt)


        check_walls(list_particles[0])
        check_walls(list_particles[1])

        detect_circle_collision(list_particles[0], list_particles[1])

        polygons[0].detect_circle_polygon_collision(list_particles[0])
        polygons[0].detect_circle_polygon_collision(list_particles[1])

        list_particles[0].move(dt)
        list_particles[1].move(dt)

        list_particles[0].draw(gui)
        list_particles[1].draw(gui)

        polygons[0].vertices_to_np()

        polygons[0].draw(gui)

        # debug(polygons[0].distances, polygons[0].vertices, gui)
        # gui.line(polygons[0].vertices[0], polygons[0].distances[0], color=0xFFFFF, radius=1)





        gui.show()


if __name__ == "__main__":
    main()
