import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size': 18})

# True Slope
x = 3
a = np.arange(-2,2,0.25)
a = a.reshape(-1,1)
b= x*a + np.random.randn(*a.shape)

plt.plot(a, x*a, color='k', linewidth=2, label='True Line')
plt.plot(a, b, 'x', color='r', markersize=10, label='Noisy data')

U, S, VT = np.linalg.svd(a, full_matrices=False)
xtilde= VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

plt.plot(a, xtilde*a, '--', color='b', linewidth=4, label='Regression Line')

plt.xlabel('a')
plt.ylabel('b')

plt.grid(linestyle='--')
plt.legend()
plt.show()
