import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.subplots()
X=np.load('3_5.npy')
Generation=np.array(X).shape[0]
Population =np.array(X).shape[1]
for i in range(Generation):
    plt.clf()
    plt.xlim(-10,110)
    plt.ylim(-0.1,1.1)
    plt.scatter(X[i,:,0],X[i,:,1])
    plt.pause(0.2)
plt.ioff()
plt.show()