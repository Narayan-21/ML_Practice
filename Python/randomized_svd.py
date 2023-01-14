import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

plt.rcParams['figure.figsize'] = [32,12]
plt.rcParams.update({'font.size': 18})

def rSVD(X, r, q, p):
    # q = number of power iterations
    # p = over-sampling factor
    ny = X.shape[1]
    P = np.random.randn(ny, r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)
    Q, R = np.linalg.qr(Z, mode='reduced')

    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY

    return U, S, VT

A = imread('jupiter.jpg')
X = np.mean(A, axis=2)

U,S,VT = np.linalg.svd(X, full_matrices=0)

r=100
q=1
p=5

rU,rS,rVT = rSVD(X,r,q,p)

# Reconstruction
XSVD = U[:,:(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:]
errSVD = np.linalg.norm(X-XSVD, ord=2)/ np.linalg.norm(X, ord=2)

XrSVD = rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:]
errSVD = np.linalg.norm(X-XrSVD,ord=2)/np.linalg.norm(X, ord=2)

fig, axs = plt.subplots(1,3)
plt.set_cmap('gray')
axs[0].imshow(X)
axs[0].axis('off')
axs[1].imshow(XSVD)
axs[1].axis('off')
axs[2].imshow(XrSVD)
axs[2].axis('off')
plt.show()
