# Understanding the essence of PCA using an example.

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16,8]

# Center of Mean/data
xC = np.array([2,1])

# Pricipal Axes
sig = np.array([2,0.5])

# Rotation of the cloud by pi/3
theta = np.pi/3

# Rotation Matrix
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

nPoints = 10000
X = R @ np.diag(sig) @ np.random.randn(2, nPoints) + np.diag(xC) @ np.ones((2, nPoints))
print('X =', X)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[0,:], X[1,:],'.', color='k')
ax1.grid()
plt.xlim((-6,8))
plt.ylim((-6,8))

Xavg = np.mean(X, axis=1)
B = X - np.tile(Xavg, (nPoints,1)).T
print('Xavg =',Xavg)
print('B = ',B)

U, S, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices = 0)
ax2 = fig.add_subplot(122)
ax2.plot(X[0,:], X[1,:],'.', color='k')
ax2.grid()
plt.xlim((-6,8))
plt.ylim((-6,8))

theta = 2*np.pi*np.arange(0,1,0.01)

# 1-std confidence interval
Xstd = U @ np.diag(S) @ np.array([np.cos(theta), np.sin(theta)])
print('Xstd=',Xstd)

ax2.plot(Xavg[0] + Xstd[0,:], Xavg[1] + Xstd[1,:], '-', color='r', linewidth=3)
ax2.plot(Xavg[0] + 2*Xstd[0,:], Xavg[1] + 2*Xstd[1,:], '-', color='r', linewidth=3)
ax2.plot(Xavg[0] + 3*Xstd[0,:], Xavg[1] + 3*Xstd[1,:], '-', color='r', linewidth=3)

ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,0]*S[0]]),
         np.array([Xavg[1], Xavg[1]+U[1,0]*S[0]]), '-', color='cyan', linewidth=5)
ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,1]*S[1]]),
         np.array([Xavg[1], Xavg[1]+U[1,1]*S[1]]), '-', color='cyan', linewidth=5)
plt.show()
