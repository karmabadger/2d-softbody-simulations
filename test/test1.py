# fractal.py
import taichi as ti
import numpy as np

import particle as p
import time

ti.init(arch=ti.cuda)

width = 1000
height = 1000


tmp = ti.Vector.field(2, dtype=ti.f32, shape=(2,))


# a_pos = ti.Vector.field(2, dtype=ti.f32, shape=a_np.shape)
p1 = p.Blue(2, m=1000, r=2.0)


p2 = p.Blue(2, m=1000000, r=2.0)

p1.set_position(0.1, 0.1)
p2.set_position(0.5, 0.5)


# a_pos.from_numpy(a_np)

# @ti.kernel
# def initialize():
#     a_pos.position[0][0] = 0.0
#     a_pos.position[0][1] = 0


# @ti.func
# def next_step():
#     a_particle.position[0] += 0.001
#     a_particle.position[1] += 0.001


@ti.pyfunc
def check_walls(particle):
    if particle.position[0].x >= 1 or particle.position[0].x <= 0:
        particle.velocity[0].x = -particle.velocity[0].x
    if particle.position[0].y >= 1 or particle.position[0].y <= 0:
        particle.velocity[0].y = -particle.velocity[0].y

@ti.pyfunc
def clear_forces():

    p1.set_force(0.0, 0.0)
    p2.set_force(0.0, 0.0)



@ti.kernel
def apply_g():
    # for i in ti.static(range(0, p1.dim)):
    #     tmp[0][i] = p1.position[0][i] - p2.position[0][i]

    tmp[0].x = p1.position[0].x - p2.position[0].x
    tmp[0].y = p1.position[0].y - p2.position[0].y
    r2 = tmp[0].norm_sqr()
    g = (p.G * ((p1.mass * p2.mass) / r2))

    p1.force[0] -= g * tmp[0].normalized()
    p2.force[0] += g * tmp[0].normalized()


@ti.pyfunc
def mag(v):
    magnitude = ti.sqrt(v[0].x ** 2 + v[0].y ** 2)
    return magnitude


@ti.pyfunc
def debug(vector, position, gui):
    tmp[0].x = position[0].x + vector[0].x
    tmp[0].y = position[0].y + vector[0].y
    gui.line(position[0], tmp[0], color = 0xFFFFFF, radius = 1)


@ti.pyfunc
def main():

    gui = ti.GUI("test", res=(width, height))

    old_t = time.time()
    new_t = time.time()

    p1.set_velocity(0.00000, 0.0004)

    for i in range(1000000):
        old_t = new_t
        new_t = time.time()
        dt = new_t - old_t


        clear_forces()
        apply_g()



        # for i in range(0, p1.dim):
        #     tmp[0][i] = p1.position[0][i] - p2.position[0][i]

        # tmp.x[0] = p1.position.x[0]  - p2.position.x[0]
        # # tmp[0].y = p1.position[0].y - p2.position[0].y
        # r2 = tmp.norm_sqr()
        # g = (particle.G * ((p1.mass * p2.mass) / r2))
        # p1.force[0] += g
        # p2.force[0] += g

        p1.apply_force(dt)
        check_walls(p1)
        p1.move(dt)
        p1.draw(gui)

        p2.apply_force(dt)
        check_walls(p2)
        p2.move(dt)
        p2.draw(gui)
        debug(p1.force, p1.position, gui)

        # gui.circle(a_particle.position[0], radius=a_particle.radius, color=0x4FB99F)
        # gui.circle(b_pos[0], radius=5, color=0xFFFFFF)
        gui.show()


if __name__ == "__main__":
    main()
