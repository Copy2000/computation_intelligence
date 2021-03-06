import numpy as np
import matplotlib.pyplot as plt
from sko.PSO import PSO
from matplotlib.animation import FuncAnimation

def demo_func(x):
    x1, x2 = x
    return x1 ** 2 + (x2 - 0.05) ** 2

pso = PSO(func=demo_func, dim=2, pop=20, max_iter=150, lb=[-1, -1], ub=[1, 1])
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

record_value = pso.record_value
X_list, V_list = record_value['X'], record_value['V']

fig, ax = plt.subplots(1, 1)
ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 20), np.linspace(-1.0, 1.0, 20))
Z_grid = demo_func((X_grid, Y_grid))
ax.contour(X_grid, Y_grid, Z_grid, 20)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.ion()
p = plt.show()


def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line


ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=300)
# plt.show()
# ani.save('test.mp4',fps=25)

ani.save('pso.gif',writer='pillow')
