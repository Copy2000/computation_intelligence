from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

frame_count = 0
points = reading_file("some_data") # this method is not of intrest

def make_one_point(i):
    global frame_count, points

    ex = [1]
    ey = [1]
    ez = [1]
    point = points[i]
    frame = point[frame_count]
    ex[0] = frame[0]
    ey[0] = frame[1]
    ez[0] = frame[2]
    frame_count += 1

    return ex, ey, ez

def update(i):
    global frame_count, points

    if frame_count < len(points[i]):
        return make_one_point(i)
    else:
        frame_count = 0
        return make_one_point(i)


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlim3d(-500, 2000)
ax1.set_ylim3d(-500, 2000)
ax1.set_zlim3d(0, 2000)

x = [1]
y = [1]
z = [1]
scat = ax1.scatter(x,y,z)

def animate(i):
    scat._offsets3d = update(0)

ani = animation.FuncAnimation(fig, animate, 
    frames=len(points[10]),
    interval=100, repeat=True)

plt.show()