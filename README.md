# IFM O3R225 Camera Integration

This repository provides tools to interface with the **IFM O3R225** 3D camera, allowing real-time visualization of 3D point clouds using **Open3D** and **IFM3D**.

## Features
- Capture 3D point cloud data from the IFM O3R225 camera.
- Real-time visualization of point clouds using Open3D.
- Handles camera connection and data retrieval for both 3D and 2D channels.

## Requirements

### Hardware
- **IFM O3R225** 3D Camera.
- Proper network setup to communicate with the camera (ensure the IP and ports are configured).

### Software
- Python 3.8+
- [Open3D](http://www.open3d.org/) (for 3D visualization)
- [IFM3D](https://github.com/ifm/ifm3d) (for camera communication)
- [NumPy](https://numpy.org/) (for array manipulation)

### References
- ArUco marker: https://www.youtube.com/watch?v=bS00Vs09Upw
