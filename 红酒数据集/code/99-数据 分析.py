import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
X_list = np.load("3_5.npy")
fig, ax = plt.subplots(1, 1)
ax.set_title('10_10', loc='center')
line = ax.plot([], [], 'b.')
ax.set_xlim(-10, 110)
ax.set_ylim(-0.1, 1.1)
plt.ion()
p = plt.show()
def update_scatter(frame):
    i, j = frame // 5, frame % 5
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i]
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line
ani = FuncAnimation(fig, update_scatter, blit=True, interval=50, frames=300)
ani.save('pso.gif', writer='pillow')