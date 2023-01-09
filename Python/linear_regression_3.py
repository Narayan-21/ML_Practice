import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size': 18})

H = np.loadtxt('housing.data')
b = H[:, -1] # Housing values in $1000s
A = H[:, :-1] # Other factors

# Pad with ones for nonzero offset
A = np.pad(A, [(0,0),(0,1)], mode='constant', constant_values=1)

n=253
btrain = b[1:n]
Atrain = A[1:n]
btest = b[n:]
Atest = A[n:]

U,S,VT = np.linalg.svd(Atrain, full_matrices=0)
X = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ btrain

fig = plt.figure()
ax1 = fig.add_subplot(121)

plt.plot(btrain, color='k', linewidth=2, label = 'Housing Value')
plt.plot(Atrain@X, '-o', color='r', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value [$1k]')
plt.legend()

ax2 = fig.add_subplot(122)
plt.plot(btest, color='k', linewidth=2, label='Housing Value')
plt.plot(Atest@X, '-o', color='r', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.legend()
plt.show()

# Analyzing the significance of different attributes contributing to the housing price.
A_mean = np.mean(A, axis=0)
A_mean = A_mean.reshape(-1,1)

A2 = A - np.ones((A.shape[0], 1)) @ A_mean.T
for j in range(A.shape[1]-1):
    A2std = np.std(A2[:,j])
    A2[:,j] = A2[:,j]/A2std
A2[:,-1] = np.ones(A.shape[0])

U,S,VT = np.linalg.svd(A2, full_matrices=0)
X = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
x_tick = range(len(X)-1) + np.ones(len(X)-1)
plt.bar(x_tick, X[:-1])
plt.xlabel('Attribute')
plt.ylabel('Significance')
plt.xticks(x_tick)
plt.show()
