# ArUco Marker Detection with IFM3D Camera

This project demonstrates how to use the IFM O3R225 3D camera to detect ArUco markers in real-time and compute their 3D coordinates using OpenCV and the IFM3D Python library.

## Features

- Real-time detection of ArUco markers.
- Estimation of the 3D position of detected markers.
- Distance measurement from the camera to the markers.
- Visualization of RGB and distance images with markers and their coordinates.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- IFM3D Python library

You can install the required libraries using pip:

```bash
pip install opencv-python numpy ifm3dpy
```
## References

    ArUco marker: (https://www.youtube.com/watch?v=bS00Vs09Upw, https://pure.tue.nl/ws/portalfiles/portal/212917849/1253174_Accuracy_of_Single_Camera_Pose_Estimation.pdf)
