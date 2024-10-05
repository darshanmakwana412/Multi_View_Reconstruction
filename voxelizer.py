import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Volume:
    def __init__(self, origin, size, n_voxels):
        self.origin = origin  # (a, b, c)
        self.size = size      # (h, w, d)
        self.n_voxels = n_voxels  # (nx, ny, nz)
        self.voxel_size = (
            size[0] / n_voxels[0],
            size[1] / n_voxels[1],
            size[2] / n_voxels[2]
        )
        self.weights = np.zeros(self.n_voxels, dtype=float)

    def get_voxel(self, x, y, z):
        a, b, c = self.origin
        h, w, d = self.size

        if not (a <= x <= a + h and b <= y <= b + w and c <= z <= c + d):
            raise ValueError("Point is outside the volume")

        i = int((x - a) / self.voxel_size[0])
        j = int((y - b) / self.voxel_size[1])
        k = int((z - c) / self.voxel_size[2])

        i = min(i, self.n_voxels[0] - 1)
        j = min(j, self.n_voxels[1] - 1)
        k = min(k, self.n_voxels[2] - 1)

        return (i, j, k)

    def add_weight(self, x, y, z, weight):
        i, j, k = self.get_voxel(x, y, z)
        self.weights[i, j, k] += weight

    def ray_cast(self, p1, dirc, n_steps=25):
        points = []
        voxels = []

        a, b, c = self.origin
        h, w, d = self.size
        voxel_size_x, voxel_size_y, voxel_size_z = self.voxel_size
        nx, ny, nz = self.n_voxels
        dirc = dirc / np.linalg.norm(dirc)

        # Check if the starting point is inside the volume
        if not (a <= p1[0] <= a + h and b <= p1[1] <= b + w and c <= p1[2] <= c + d):
            raise ValueError("Starting point is outside the volume")

        # Get the current voxel indices
        i, j, k = self.get_voxel(p1[0], p1[1], p1[2])

        # Compute the step direction
        stepX = 1 if dirc[0] > 0 else -1
        stepY = 1 if dirc[1] > 0 else -1
        stepZ = 1 if dirc[2] > 0 else -1

        # Compute initial tMax and tDelta for each axis
        if dirc[0] != 0:
            if dirc[0] > 0:
                next_voxel_boundary_x = a + (i + 1) * voxel_size_x
            else:
                next_voxel_boundary_x = a + i * voxel_size_x
            tMaxX = (next_voxel_boundary_x - p1[0]) / dirc[0]
            tDeltaX = voxel_size_x / abs(dirc[0])
        else:
            tMaxX = float('inf')
            tDeltaX = float('inf')

        if dirc[1] != 0:
            if dirc[1] > 0:
                next_voxel_boundary_y = b + (j + 1) * voxel_size_y
            else:
                next_voxel_boundary_y = b + j * voxel_size_y
            tMaxY = (next_voxel_boundary_y - p1[1]) / dirc[1]
            tDeltaY = voxel_size_y / abs(dirc[1])
        else:
            tMaxY = float('inf')
            tDeltaY = float('inf')

        if dirc[2] != 0:
            if dirc[2] > 0:
                next_voxel_boundary_z = c + (k + 1) * voxel_size_z
            else:
                next_voxel_boundary_z = c + k * voxel_size_z
            tMaxZ = (next_voxel_boundary_z - p1[2]) / dirc[2]
            tDeltaZ = voxel_size_z / abs(dirc[2])
        else:
            tMaxZ = float('inf')
            tDeltaZ = float('inf')

        t = 0
        current_voxel = (i, j, k)
        current_point = p1.copy()

        for step in range(n_steps):
            voxels.append(current_voxel)
            points.append(current_point.copy())

            # Determine the next voxel to step into
            if tMaxX < tMaxY:
                if tMaxX < tMaxZ:
                    # Step in x direction
                    t = tMaxX
                    i += stepX
                    tMaxX += tDeltaX
                else:
                    # Step in z direction
                    t = tMaxZ
                    k += stepZ
                    tMaxZ += tDeltaZ
            else:
                if tMaxY < tMaxZ:
                    # Step in y direction
                    t = tMaxY
                    j += stepY
                    tMaxY += tDeltaY
                else:
                    # Step in z direction
                    t = tMaxZ
                    k += stepZ
                    tMaxZ += tDeltaZ

            # Update current point
            current_point = p1 + dirc * t

            # Check if the new indices are within the volume
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                print(f"Ray has exited the volume at step {step}")
                break
            current_voxel = (i, j, k)

        return points, voxels

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        a, b, c = self.origin
        h, w, d = self.size
        x = [a, a + h]
        y = [b, b + w]
        z = [c, c + d]
        for s, e in combinations(np.array(list(product(x, y, z))), 2):
            if np.sum(np.abs(s - e)) in [h, w, d]:
                ax.plot3D(*zip(s, e), color="k")

        filled = self.weights > 0
        if np.any(filled):
            nx, ny, nz = self.n_voxels
            x_edges = np.linspace(a, a + h, nx + 1)
            y_edges = np.linspace(b, b + w, ny + 1)
            z_edges = np.linspace(c, c + d, nz + 1)
            x, y, z = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')
            ax.voxels(x, y, z, filled, facecolors='red', edgecolor='k', alpha=0.5)
        else:
            print("No non-zero weight voxels to display.")

        ax.set_xlim(a, a + h)
        ax.set_ylim(b, b + w)
        ax.set_zlim(c, c + d)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()