# Creating the noisy known data and truncating the svd of that data using gavish-donoho method.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size': 18})

t = np.arange(-3,3,0.01)

Utrue = np.array([np.cos(17*t) * np.exp(-t**2), np.sin(11*t)]).T
Strue = np.array([[2,0],[0,0.5]])
Vtrue = np.array([np.sin(5*t) * np.exp(-t**2), np.cos(13*t)]).T

X = Utrue @ Strue @ Vtrue.T

sigma = 1
Xnoisy = X + sigma*np.random.randn(*X.shape) # Manually created the noisy data

U, S, VT = np.linalg.svd(Xnoisy, full_matrices=0)
N = Xnoisy.shape[0]
cutoff = (4/np.sqrt(3)) * np.sqrt(N) * sigma # Threshold using gavish method
r = np.max(np.where(S > cutoff))

Xclean = U[:, :(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:]

fig = plt.figure()
ax1 = plt.subplot(131)
ax1.imshow(X)

ax2 = plt.subplot(132)
ax2.imshow(Xnoisy)

ax3 = plt.subplot(133)
ax3.imshow(Xclean)

plt.set_cmap('gray')
plt.show()
