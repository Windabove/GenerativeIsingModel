import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

data, target = fetch_openml('mnist_784', return_X_y=True, as_frame=False)

tar = '3'
beta = 100
eta = 0.01
rr = 0.995
M = 3
U = 1

D = (data[target == tar] > 125).astype(np.float32) * 2 - 1
J = D.T.dot(D) / D.shape[0]
for i in range(784):
    J[i, i] = 0
    m1, n1 = i // 28, i % 28
    for j in range(784):
        m2, n2 = j // 28, j % 28
        if (m2 - m1 + 28) % 28 + (n2 - n1 + 28) % 28 > 6:
            J[i, j] = 0

B = D.mean(0)

grid = np.random.randint(0, 2, 784) * 2 - 1
totalE = (grid.dot(U * J) + M * B).dot(grid)
refused = 0

grids = []
energy = []

for i in range(1000001):
    ind = np.random.randint(0, 784)
    deltaE = 2 * (grid.dot(U * J[ind]) + M * B[ind]) * grid[ind]
    if i % 1000 == 0:
        refused = 0
    if (deltaE < 0 or np.random.rand() < np.exp(- deltaE / beta)):
        grid[ind] *= -1
        totalE += deltaE
        if beta > 0.1:
            beta *= 1 - eta
        if i % 100 == 0:
            print(totalE)
    else:
        refused += 1
    if i % 10 == 0:
        grids.append(grid.reshape(28, 28).copy())
        energy.append(totalE)
    if refused / 1000 > rr:
        print("Stopped at step {}".format(i))
        grids.append(grid.reshape(28, 28))
        break

plt.imshow(grids[-1])
plt.show()

# matplotlib.use('Agg')
# for i in range(len(grids)):
#     fig, axes = plt.subplots(1, 2, figsize=(15, 15))
#     result = grids[i]
#     axes[0].set_xticks([])
#     axes[0].set_yticks([])
#     axes[0].imshow(result, cmap='gray')
#     axes[1].set_xlabel('steps')
#     axes[1].set_ylabel('total energy')
#     axes[1].plot(energy[:i])
#     plt.savefig('{}/{}.png'.format(tar, i))
#     plt.close(fig)
