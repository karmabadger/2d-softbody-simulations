# fractal.py
import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

n = 500

a_np = np.array([0.2,0.5], dtype='f')
b_np = np.array([0.8,0.5], dtype='f')


a_pos = ti.Vector.field(2, dtype=ti.f32, shape=a_np.shape)
b_pos = ti.Vector.field(2, dtype=ti.f32, shape=b_np.shape)
# a_pos = ti.Vector([0.2,0.5])
# b_pos = ti.Vector([0.8,0.5])


# a_pos.from_numpy(a_np)

@ti.kernel
def initialize():
    a_pos[0][0] = 0.0
    a_pos[0][1] = 0
    b_pos[0][0] = 0
    b_pos[0][1] = 0
    # a_pos.fill(0)



@ti.func
def next_step():
    a_pos[0] += 0.001
    a_pos[1] += 0.001


@ti.kernel
def run():
    next_step()


    

initialize()
gui = ti.GUI("test", res=(n, n))



for i in range(1000000):
    run()
    gui.circle(a_pos[0], radius=5, color=0x4FB99F)
    gui.circle(b_pos[0], radius=5, color=0xFFFFFF)
    gui.show()
