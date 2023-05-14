import taichi as ti
# import numpy as np


@ti.data_oriented
class Polygon:
    """Polygons

    """

    number_particles = 0

    def __init__(self, dim, pinned=False, p_vertices=[], color=0xFFFFFF, thickness=1.0):
        # print("hello")
        self.dim = dim
        self.pid = Polygon.number_particles
        Polygon.number_particles += 1
        # self.position = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        # self.velocity = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        # self.force = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        # self.distance = ti.Vector.field(dim, dtype=ti.f32, shape=(1,))
        # self.mass = mass

        # line thickness
        self.radius = thickness
        self.color = color
        self.pinned = pinned

        self.type = 'polygon'

        self.number_vertices = len(p_vertices)
        # self.np_begin = np.zeros((self.number_vertices,))
        # self.np_end = np.zeros((self.number_vertices,))

        # self.i_count = 0
        # self.distances = ti.Vector.field(dim, dtype=ti.f32, shape=(self.number_vertices,))
        # self.edges = ti.Vector.field(dim, dtype=ti.f32, shape=(self.number_vertices,))
        # self.normals = ti.Vector.field(dim, dtype=ti.f32, shape=(self.number_vertices,))
        # self.change = ti.Vector.field(dim, dtype=ti.f32, shape=(self.number_vertices,))
        # self.tmp = ti.Vector.field(dim, dtype=ti.f32, shape=(self.number_vertices,))
        self.shortest_ind = ti.field(dtype=ti.i32, shape=())

        self.is_collided = ti.field(dtype=ti.f32, shape=())

        self.vertices = ti.Vector.field(dim, dtype=ti.f32, shape=(4,))
        for i in range(len(p_vertices)):
            self.vertices[i].x = p_vertices[i][0]
            self.vertices[i].y = p_vertices[i][1]
            # print(self.vertices[i].x)

        # self.shortest_ind = 0





    @ti.pyfunc
    def draw(self, gui):

        # self.debug(self.normals[self.shortest_ind], self.position, gui)
        # draw normals
        # gui.line(self.vertices[self.shortest_ind[None]], self.distances[self.shortest_ind[None]],color=0xFFFFF, radius=1 )

        # pos2 = ti.Vector([self.vertices[0].x + self.edges[0].x, self.vertices[0].y + self.edges[0].y])
        # gui.line(self.vertices[0], pos2, color=0xFFFFF, radius=1)
        # gui.circle(pos2, radius=10, color=0xFF4444)
        #
        # gui.line(self.vertices[0], self.position[0], color=0xFFFFF, radius=1)
        #
        # pos2 = ti.Vector([self.vertices[0].x + self.normals[0].x, self.vertices[0].y + self.normals[0].y])
        # gui.line(self.vertices[0], pos2, color=0x00FF00, radius=1)
        #
        # pos2 = ti.Vector([self.position[0].x + self.change[0].x, self.position[0].y + self.edges[0].y])
        # gui.line(self.vertices[0], pos2, color=0xFFFF00, radius=1)

        # print("v: ", self.vertices[0].x, ", ", self.vertices[0].y, " d: ", self.distances[0].x, ", ",
        #       self.distances[0].y)
        # gui.circle(self.position[0], radius=100, color=0xFF4444)
        # gui.line(self.vertices[1], self.position[0], color=0xFFFFF, radius=1)
        # gui.line(self.vertices[2], self.position[0], color=0xFFFFF, radius=1)
        # gui.line(self.vertices[3], self.position[0], color=0xFFFFF, radius=1)
        # gui.line(self.vertices[0], self.normals[self.shortest_ind[None]],color=0xFFFFF, radius=1 )

        for i in range(self.number_vertices):
            gui.line(self.vertices[i], self.vertices[(i + 1) % self.number_vertices], color=self.color,
                     radius=self.radius)

    @ti.pyfunc
    def vertices_to_np(self):
        for i in range(self.number_vertices):
            self.np_begin[i] = self.vertices[i].x
            self.np_end[i] = self.vertices[i].y

    # def inc_count_i(self):
    #     self.i_count += 1

    # @ti.kernel
    # def detect_circle_polygon_collision(self, p: ti.template()):
    #
    #     print("p: ", p.position[0])
    #
    #     bbox = self.check_bounding_box(p)
    #
    #     # bbox = 1
    #     # if bbox != 0:
    #
    #     is_inside = self.check_same_side(p.position)
    #
    #     if bbox != 0 or is_inside != 0:
    #
    #         # collided with bounding box
    #         #     print("collide i: ", self.i_count)
    #         #     self.inc_count_i()
    #
    #         distances_mag = []
    #
    #         for i in range(self.number_vertices):
    #             self.distances[i] = p.position[0] - self.vertices[i]
    #             distances_mag.append(self.distances[i].norm())
    #
    #         self.position[0] = (self.vertices[0] + self.distances[0])
    #
    #         for i in range(self.number_vertices):
    #             self.edges[i] = self.vertices[(i + 1) % self.number_vertices] - self.vertices[i]
    #
    #         normals_mag = []
    #         for i in ti.static(range(self.number_vertices)):
    #             dot = self.distances[i].dot(self.edges[i])
    #             d2 = self.edges[i].dot(self.edges[i])
    #             dot_div_2 = dot / d2
    #             dot_vec = dot_div_2 * self.edges[i]
    #             # if is_inside != 0:
    #             #     print("is in")
    #             #     self.normals[i] = (dot_vec - self.distances[i]).normalized()
    #             #     old_pos = p.position[0] - p.velocity[0]
    #             #     touch_point = self.line_intersection(old_pos, p.position[0], self.vertices[self.shortest_ind],
    #             #                                          self.vertices[self.shortest_ind + 1])
    #             #
    #             #     diff = p.position[0] - touch_point
    #             #
    #             #     dot_prod = diff.dot(self.normals[self.shortest_ind])
    #             #
    #             #     proj_normal = dot_prod * self.edges[self.shortest_ind]
    #             #
    #             #     p.position[0] -= 2* proj_normal
    #             #
    #             #     p.velocity[0] = ti.Vector([0,0])
    #             #
    #             #     self.is_collided[None] = 1
    #             # else:
    #             #     self.normals[i] = (self.distances[i] - dot_vec).normalized()
    #             self.normals[i] = (self.distances[i] - dot_vec).normalized()
    #             normals_mag.append(self.normals[i].norm_sqr())
    #
    #
    #         self.shortest_ind[None] = 0
    #         shortest_dist = normals_mag[0]
    #         for unknown in range(1):
    #             for i in ti.static(range(self.number_vertices)):
    #                 if shortest_dist < normals_mag[i]:
    #                     shortest_dist = normals_mag[i]
    #                     self.shortest_ind[None] = i
    #
    #         # bounce
    #         change = p.velocity[0].dot(self.normals[self.shortest_ind])
    #         # if is_inside == 0:
    #         #     change = -change
    #
    #         self.change[0] = change * self.normals[self.shortest_ind]
    #
    #         # p.velocity[0] -= 1.6 * self.change[0]
    #         if self.is_collided[None] == 0:
    #             # p.velocity[0] -= 1.6 * self.change[0]
    #             # p.velocity[0] -= 0.6 * self.change[0]
    #
    #             self.is_collided[None] = 1
    #
    #         else:
    #             print("can't collide")
    #             self.is_collided[None] = 0
    #
    #     else:
    #         print("not colliding")
    #         self.is_collided[None] = 0
    #
    # @ti.kernel
    # def circle_polygon_collision_force(self, p: ti.template()):
    #
    #     print("p: ", p.position[0])
    #
    #     bbox = self.check_bounding_box(p)
    #
    #     # bbox = 1
    #     # if bbox != 0:
    #
    #     is_inside = self.check_same_side(p.position)
    #
    #     if bbox != 0 or is_inside != 0:
    #
    #         # collided with bounding box
    #         #     print("collide i: ", self.i_count)
    #         #     self.inc_count_i()
    #
    #         distances_mag = []
    #
    #         for i in range(self.number_vertices):
    #             self.distances[i] = p.position[0] - self.vertices[i]
    #             distances_mag.append(self.distances[i].norm())
    #
    #         self.position[0] = (self.vertices[0] + self.distances[0])
    #
    #         for i in range(self.number_vertices):
    #             self.edges[i] = self.vertices[(i + 1) % self.number_vertices] - self.vertices[i]
    #
    #         normals_mag = []
    #         for i in ti.static(range(self.number_vertices)):
    #             dot = self.distances[i].dot(self.edges[i])
    #             d2 = self.edges[i].dot(self.edges[i])
    #             dot_div_2 = dot / d2
    #             dot_vec = dot_div_2 * self.edges[i]
    #             if is_inside != 0:
    #                 print("is in")
    #                 self.normals[i] = (dot_vec - self.distances[i]).normalized()
    #                 old_pos = p.position[0] - p.velocity[0]
    #                 touch_point = self.line_intersection(old_pos, p.position[0], self.vertices[self.shortest_ind],
    #                                                      self.vertices[self.shortest_ind + 1])
    #
    #                 diff = p.position[0] - touch_point
    #
    #                 dot_prod = diff.dot(self.normals[self.shortest_ind])
    #
    #                 proj_normal = dot_prod * self.edges[self.shortest_ind]
    #
    #                 p.position[0] -= 2 * proj_normal
    #
    #                 p.velocity[0] = ti.Vector([0, 0])
    #
    #                 self.is_collided[None] = 1
    #             else:
    #                 self.normals[i] = (self.distances[i] - dot_vec).normalized()
    #
    #             normals_mag.append(self.normals[i].norm_sqr())
    #
    #         self.shortest_ind[None] = 0
    #         shortest_dist = normals_mag[0]
    #         for unknown in range(1):
    #             for i in ti.static(range(self.number_vertices)):
    #                 if shortest_dist < normals_mag[i]:
    #                     shortest_dist = normals_mag[i]
    #                     self.shortest_ind[None] = i
    #
    #         # bounce
    #         change = p.velocity[0].dot(self.normals[self.shortest_ind])
    #         # if is_inside == 0:
    #         #     change = -change
    #
    #         self.change[0] = change * self.normals[self.shortest_ind]
    #
    #         # p.velocity[0] -= 1.6 * self.change[0]
    #         if self.is_collided[None] == 0:
    #             p.velocity[0] -= 1.6 * self.change[0]
    #             # p.velocity[0] -= 0.6 * self.change[0]
    #
    #             self.is_collided[None] = 1
    #
    #         else:
    #             print("can't collide")
    #             self.is_collided[None] = 0
    #
    #     else:
    #         self.is_collided[None] = 0

    # @ti.pyfunc
    # def check_same_side(self, r):
    #     is_same = 1
    #
    #     side = -1
    #
    #     p = 0
    #     q = 1
    #     orient = self.orient(self.vertices[p], self.vertices[q], r)
    #
    #     # print("p: ", self.vertices[p], " diff: ",
    #     #       self.orient(self.vertices[p], self.vertices[q], r),
    #     #       ", q:", self.vertices[q],
    #     #       ", r:", r[0])
    #
    #     for i in ti.static(range(1, self.number_vertices)):
    #         p = i
    #         q = (i + 1) % self.number_vertices
    #
    #         orient2 = self.orient(self.vertices[p], self.vertices[q], r)
    #
    #         if orient != orient2:
    #             # print(orient2[0])
    #             is_same = 0
    #
    #     return is_same
    #
    # @ti.pyfunc
    # def check_bounding_box(self, p):
    #
    #     is_in = 1.0
    #
    #     bound = ti.Vector([0.0, 0.0, 0.0, 0.0])
    #     bound[0] = self.vertices[0].x
    #     bound[1] = self.vertices[0].x
    #     bound[2] = self.vertices[0].y
    #     bound[3] = self.vertices[0].y
    #
    #     for j in range(1):
    #         for i in range(self.number_vertices):
    #             vertex = self.vertices[i]
    #             bound[0] = min(bound[0], vertex[0])
    #             bound[1] = max(bound[1], vertex[0])
    #             bound[2] = min(bound[2], vertex[1])
    #             bound[3] = max(bound[3], vertex[1])
    #             # bound[0] = self.minres(bound[0], vertex[0])
    #             # bound[1] = self.maxres(bound[1], vertex[0])
    #             # bound[2] = self.minres(bound[2], vertex[1])
    #             # bound[3] = self.maxres(bound[3], vertex[1])
    #
    #         # print("radius: " , p.radius)
    #
    #         if p.position[0].x + p.r_radius < bound[0] or p.position[0].x - p.r_radius > bound[1] or p.position[
    #             0].y + p.r_radius < bound[2] or p.position[0].y - p.r_radius > bound[3]:
    #             is_in = 0.0
    #
    #     return is_in

    # @ti.pyfunc
    # def minres(self, x1: ti.f32, x2: ti.f32):
    #     minx = x2
    #     if x1 < x2:
    #         minx = x1
    #     return minx

    # @ti.pyfunc
    # def maxres(self, x1: ti.f32, x2: ti.f32):
    #     maxx = x2
    #     if x1 > x2:
    #         maxx = x1
    #     return maxx

    """
    Intersection point between 2 lines each defined by 2 points
    """

    @ti.func
    def line_intersection(self, p0, p1, q0, q1):
        D = (p0.x - p1.x) * (q0.y - q1.y) - (p0.y - p1.y) * (q0.x - q1.x)

        linep = ((p0.x * p1.y - p0.y * p1.x) * (q0.x - q1.x) - (p0.x - p1.x) * (q0.x * q1.y - q0.y * q1.x)) / D
        lineq = ((p0.x * p1.y - p0.y * p1.x) * (q0.y - q1.y) - (p0.y - p1.y) * (q0.x * q1.y - q0.y * q1.x)) / D

        vec = ti.Vector([linep, lineq])

        return vec

    @ti.func
    def on_segment(self, p, q, r):
        is_on_seg = false

        if max(p.x, r[0].x) >= q.x >= min(p.x, r[0].x) and max(p.y, r[0].y) >= q.y >= min(p.y, r[0].y):
            is_on_seg = true

        return is_on_seg

    @ti.func
    def orient(self, p, q, r):
        orient_dir = (q.y - p.y) * (r[0].x - q.x) - (q.x - p.x) * (r[0].y - q.y)

        # print(
        #     " o:", orient_dir)
        if orient_dir > 0:
            orient_dir = 1
        elif orient_dir < 0:
            orient_dir = -1
        elif orient_dir == 0:
            orient_dir == 0

        return orient_dir

    # @ti.func
    # def mult(self, m1: ti.f32, m2: ti.f32) -> ti.f32:
    #     m = m1 * m2
    #     return m
