import math

class Vec3D:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def zero():
        return Vec3D(0, 0, 0)

    @staticmethod
    def one():
        return Vec3D(1, 1, 1)

    def add(self, vec: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x + vec.x, self.y + vec.y, self.z + vec.z)

    def sub(self, vec: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x - vec.x, self.y - vec.y, self.z - vec.z)

    def scale(self, s: float) -> 'Vec3D':
        return Vec3D(self.x * s, self.y * s, self.z * s)
    
    def norm(self):
        r = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        return Vec3D(self.x / r, self.y / r, self.z / r)

    def __str__(self):
        return f"Vec3D(x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return f"Vec3D(x={self.x}, y={self.y}, z={self.z})"
    
class VisualHull:
    def __init__(self, resolution: int = 64):
        self.resolution = resolution
        self.volume = np.ones((resolution, resolution, resolution), dtype=np.float32)
        
    def reconstruct(self, scene: 'Scene', num_samples: int) -> np.ndarray:
        for i in range(self.num_samples):
            _, pose, mask = scene.sample()
            self.carve_volume(mask, pose)
        return self.volume
    
    def carve_volume(self, mask: np.ndarray, camera_pose: np.ndarray) -> None:
        mask_points = np.argwhere(mask > 0)
        for point in mask_points:
            projected_voxel = self.project_to_volume(point, camera_pose)
            if self.is_inside_volume(projected_voxel):
                self.volume[tuple(projected_voxel)] = 0.0

def plot_ray(volume, p1, dirc, n_steps=25):
    points, voxels = volume.ray_cast(p1, dirc, n_steps)

    # Plot the volume and non-zero voxels
    volume.show()

    # Now plot the ray
    fig = plt.gcf()
    ax = fig.add_subplot(projection='3d')

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color='blue', marker='o', label='Ray Steps')

    # Optionally, plot the initial direction for visualization
    t_values = np.linspace(0, np.linalg.norm(volume.size) * 1.5, 100)
    ray_line = p1[np.newaxis, :] + t_values[:, np.newaxis] * (dirc / np.linalg.norm(dirc))
    ax.plot(ray_line[:, 0], ray_line[:, 1], ray_line[:, 2], color='cyan', label='Ray Direction')

    ax.legend()
    plt.show()

def depth_to_pointcloud(K: np.ndarray, depth: np.ndarray, rgb: np.ndarray = None):
    _fx = K[0, 0]
    _fy = K[1, 1]
    _cx = K[0, 2]
    _cy = K[1, 2]

    # Mask out invalid depth
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]

    # Normalize pixel coordinates
    normalized_x = x.astype(np.float32) - _cx
    normalized_y = y.astype(np.float32) - _cy

    # Convert to world coordinates
    world_x = normalized_x * depth[y, x] / _fx
    world_y = normalized_y * depth[y, x] / _fy
    world_z = depth[y, x] - 2

    pc = np.vstack((world_x, world_y, world_z)).T

    # Assign rgb colors to points if available
    if rgb is not None:
        rgb = rgb[y, x, :]

    return pc, rgb