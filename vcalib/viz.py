import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter_3d(pcloud):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = pcloud[:, 0]
    y = pcloud[:, 1]
    z = pcloud[:, 2]

    ax.scatter(x, y, z)

    plt.show()


def arange_fixed_len(a, b, n_steps=100):
    step = (b - a) / n_steps
    return np.arange(a, b, step)


def plot_plane_3d(plane_coefs, pcloud):

    min_vals = pcloud.min(axis=0)
    max_vals = pcloud.max(axis=0)

    range_x = arange_fixed_len(min_vals[0], max_vals[0])
    range_y = arange_fixed_len(min_vals[1], max_vals[1])

    xx, yy = np.meshgrid(range_x, range_y)
    zz = plane_z(plane_coefs, xx, yy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = pcloud[:, 0]
    y = pcloud[:, 1]
    z = pcloud[:, 2]

    ax.scatter(x, y, z)
    ax.plot_surface(xx, yy, zz, alpha=0.4)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
