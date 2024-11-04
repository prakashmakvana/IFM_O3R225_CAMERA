import cv2
import numpy as np
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import threading
import time

class FrameProcessor(threading.Thread):
    def __init__(self, o3r, fg, camera_matrix, distortion_coeffs):
        super(FrameProcessor, self).__init__()
        self.o3r = o3r
        self.fg = fg
        self.running = True
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs

        # Set up the ArUco parameters and dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # Create detector parameters directly
        self.aruco_params = cv2.aruco.DetectorParameters()  # Use this if create is not available

    def run(self):
        while self.running:
            [ok, frame] = self.fg.wait_for_frame().wait_for(150)  # wait with 150ms timeout

            if ok:
                rgb = frame.get_buffer(buffer_id.JPEG_IMAGE)
                nparr = np.frombuffer(rgb, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Detect ArUco markers
                corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)

                # If markers are detected, draw them
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)

                # Display the image using OpenCV
                cv2.imshow('Frame', image)
                cv2.waitKey(1)  # Adjust waitKey value for frame rate

    def stop(self):
        self.running = False

def main():
    # Camera calibration parameters
    camera_matrix = np.array([[575.92638133, 0, 643.29092004],
                               [0, 575.57937179, 389.82093439],
                               [0, 0, 1]])
    distortion_coeffs = np.array([[-3.20332932e-01, 1.28176264e-01, 4.11432050e-04,
                                    -5.10134163e-05, -2.52065749e-02]])

    o3r = O3R("192.168.21.73")
    fg = FrameGrabber(o3r, pcic_port=50013)
    fg.start([buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.JPEG_IMAGE])

    processor = FrameProcessor(o3r, fg, camera_matrix, distortion_coeffs)
    processor.start()

    try:
        while True:
            time.sleep(1)  # Main thread can do other tasks if needed

    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        processor.stop()
        fg.stop()
        cv2.destroyAllWindows()  # Close OpenCV windows

if __name__ == "__main__":
    main()
