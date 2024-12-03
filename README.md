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
- ArUco marker: (https://www.youtube.com/watch?v=bS00Vs09Upw, https://pure.tue.nl/ws/portalfiles/portal/212917849/1253174_Accuracy_of_Single_Camera_Pose_Estimation.pdf)

- Stereo Calibration: (https://www.ni.com/docs/de-DE/bundle/ni-vision-concepts-help/page/stereo_calibration_in-depth.html?srsltid=AfmBOooXUVoYzRPKjtGWuq2LCdiaAvvwKLWu2E0Fw-QIydcWZEeFLdzg)

- Bird's Eye View Transformation: https://www.ijser.org/researchpaper/A-Simple-Birds-Eye-View-Transformation-Technique.pdf
                                  https://developer.ridgerun.com/wiki/index.php/Birds_Eye_View/Introduction/Research
                                  https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1219363/full
