import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Voxelizer:
    def __init__(self, origin, size, n_voxels, num_images):
        self.origin = origin  # (a, b, c)
        self.size = size      # (h, w, d)
        self.n_voxels = n_voxels  # (nx, ny, nz)
        self.num_images = num_images
        self.voxel_size = (
            size[0] / n_voxels[0],
            size[1] / n_voxels[1],
            size[2] / n_voxels[2]
        )
        self.weights = np.zeros((self.n_voxels[0], self.n_voxels[1], self.n_voxels[2], self.num_images), dtype=float)

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

    def reset(self):
        self.weights = np.zeros((self.n_voxels[0], self.n_voxels[1], self.n_voxels[2], self.num_images), dtype=float)

    def update_weight(self, x, y, z, img_idx):
        try:
            i, j, k = self.get_voxel(x, y, z)
            self.weights[i, j, k, img_idx] = 1
        except:
            return 0

    def ray_cast(self, p1, dirc, n_steps=25, verbose: bool = False):
        points = []
        voxels = []

        a, b, c = self.origin
        h, w, d = self.size
        voxel_size_x, voxel_size_y, voxel_size_z = self.voxel_size
        nx, ny, nz = self.n_voxels
        dirc = dirc / np.linalg.norm(dirc)

        # Compute the entry and exit points of the ray with the volume
        bounds_min = np.array([a, b, c])
        bounds_max = bounds_min + np.array([h, w, d])

        t_min = (bounds_min - p1) / dirc
        t_max = (bounds_max - p1) / dirc

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        if t_enter > t_exit or t_exit < 0:
            if verbose:
                print("Ray does not intersect the volume.")
            return points, voxels  # Empty lists

        # Adjust t_enter if it's negative (ray starts inside the volume)
        t_enter = max(t_enter, 0.0)

        # Calculate the starting point within the volume
        p_start = p1 + dirc * t_enter

        # Get the starting voxel indices
        try:
            i, j, k = self.get_voxel(p_start[0], p_start[1], p_start[2])
        except ValueError:
            if verbose:
                print("Starting point after entering volume is outside. Exiting ray_cast.")
            return points, voxels

        # Initialize tMax and tDelta
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

        # Adjust tMax values based on t_enter
        if tMaxX < t_enter:
            tMaxX += ((t_enter - tMaxX) // tDeltaX + 1) * tDeltaX
        if tMaxY < t_enter:
            tMaxY += ((t_enter - tMaxY) // tDeltaY + 1) * tDeltaY
        if tMaxZ < t_enter:
            tMaxZ += ((t_enter - tMaxZ) // tDeltaZ + 1) * tDeltaZ

        # Compute the step direction
        stepX = 1 if dirc[0] > 0 else -1
        stepY = 1 if dirc[1] > 0 else -1
        stepZ = 1 if dirc[2] > 0 else -1

        t = t_enter
        current_voxel = (i, j, k)
        current_point = p_start.copy()

        for step in range(n_steps):
            # Check if the indices are within the volume
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                if verbose:
                    print(f"Ray has exited the volume at step {step}")
                break

            voxels.append(current_voxel)
            points.append(current_point.copy())

            # Determine the next voxel to step into
            if tMaxX < tMaxY:
                if tMaxX < tMaxZ:
                    # Step in x direction
                    t = tMaxX
                    tMaxX += tDeltaX
                    i += stepX
                else:
                    # Step in z direction
                    t = tMaxZ
                    tMaxZ += tDeltaZ
                    k += stepZ
            else:
                if tMaxY < tMaxZ:
                    # Step in y direction
                    t = tMaxY
                    tMaxY += tDeltaY
                    j += stepY
                else:
                    # Step in z direction
                    t = tMaxZ
                    tMaxZ += tDeltaZ
                    k += stepZ

            # Update current point
            current_point = p1 + dirc * t

            # Update current voxel
            current_voxel = (i, j, k)

            # Check if we've reached the exit point
            if t > t_exit:
                if verbose:
                    print(f"Ray has exited the volume at step {step}")
                break

        return points, voxels

    def show(self, thresh: float = 0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the volume boundaries
        a, b, c = self.origin
        h, w, d = self.size
        x = [a, a + h]
        y = [b, b + w]
        z = [c, c + d]
        # Draw the edges of the cuboid
        for s, e in combinations(np.array(list(product(x, y, z))), 2):
            if np.sum(np.abs(s - e)) in [h, w, d]:
                ax.plot3D(*zip(s, e), color="k")

        # Plot non-zero weight voxels
        filled = np.sum(self.weights, axis=-1) > thresh
        if np.any(filled):
            nx, ny, nz = self.n_voxels
            x_edges = np.linspace(a, a + h, nx + 1)
            y_edges = np.linspace(b, b + w, ny + 1)
            z_edges = np.linspace(c, c + d, nz + 1)
            x_centers = x_edges[:-1] + self.voxel_size[0] / 2
            y_centers = y_edges[:-1] + self.voxel_size[1] / 2
            z_centers = z_edges[:-1] + self.voxel_size[2] / 2
            filled_positions = np.argwhere(filled)
            xs = x_centers[filled_positions[:, 0]]
            ys = y_centers[filled_positions[:, 1]]
            zs = z_centers[filled_positions[:, 2]]
            ax.scatter(xs, ys, zs, c='red', marker='s', s=50, alpha=0.5)
        else:
            print("No non-zero weight voxels to display.")

        ax.set_xlim(a, a + h)
        ax.set_ylim(b, b + w)
        ax.set_zlim(c, c + d)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()