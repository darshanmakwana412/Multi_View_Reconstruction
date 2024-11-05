# Multi View Reconstruction

## Tasks
- [ ] Pipeline for Imagery Collection
- [ ] Camera Parameter Estimation
   ![cam_par](./assets/cam_par.png)
   - Method 1: [Automatic Camera Recovery for closed or Open Image Sequences](https://ora.ox.ac.uk/objects/uuid:e533e8d4-d750-4e0d-a9f5-92174013764a/files/sd217qr45g)
   - Method 2: [Self-Calibration and Metric Reconstruction
in spite of Varying and Unknown Intrinsic Camera Parameter](https://people.inf.ethz.ch/~pomarc/pubs/PollefeysIJCV99.pdf)
   - Method 3: SIFT (Scale Invariant Feature Transform)
- [ ] Bundle Adjustment
- [ ] Depthmap Reconstruction(not necessary can use a graph cut based method on voxel grid)
- [ ] Volumentric Fusion
- [ ] Final Rendering

# Pose Estimation

Calibrating the camera using Aruco Board
![](./assets/aruco_board.png)

Capturing images of the object on the aruco Board
![](./assets/img_capture.gif)

Estimating the Pose of the object

![](./assets/pose1.png)  |  ![](./assets/pose2.png)
:-------------------------:|:-------------------------:

## Aruco Marker Generation and Detection
[Detection of Aruco Boards](https://docs.opencv.org/4.x/db/da9/tutorial_aruco_board_detection.html#autotoc_md1053)
[Calibration with Aruco Boards](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)


## Problem Framing

We are framing the multi-view reconstruction problem as a **convex energy minimization** problem. The following resources provide key insights into this approach and offer foundational concepts to be implemented:

- [Framing the Multi View Reconstruction problem as a convex energy minimization problem](https://cvg.cit.tum.de/_media/spezial/bib/kolev-et-al-ijcv09.pdf)
- [Multi View Reconstruction from RGB + Depth Data](https://cvg.cit.tum.de/_media/spezial/bib/steinbruecker_etal_iccv2013.pdf)

## Generating Visual Hull

![](./assets/rc1.png)  |  ![](./assets/rc2.png)
:-------------------------:|:-------------------------:
![](./assets/rc3.png)  |  ![](./assets/rc4.png)

## Reconstruction from visual Hull

25^3 Voxels             |   50^3 Voxels | 100^3 Voxels
:-------------------------:|:-------------------------:|:-------------------------:
![](./assets/naruto_25.gif)  |  ![](./assets/naruto_50.gif) | ![](./assets/naruto_100.gif)

## Graph Cut

## Additional References

Though not directly relevant, the following paper provides ideas that may be useful in our case:

- [By Example 3D Reconstruction](https://talhassner.github.io/home/projects/By_Example_Reconstruction/BP06_HASSNER_T.pdf)

## Dependencies

We will be utilizing the `pytorch_volumetric` library to work with voxelized tensors:

- GitHub Repo: [UM-ARM-Lab/pytorch_volumetric](https://github.com/UM-ARM-Lab/pytorch_volumetric)

This library handles voxel tensor operations, significantly reducing the need for custom voxelization code.

## How to Use

1. Clone this repository:
   ```bash
   git clone this repo
   ```
2. Install the necessary dependencies:
   ```bash
   bruh
   ```
