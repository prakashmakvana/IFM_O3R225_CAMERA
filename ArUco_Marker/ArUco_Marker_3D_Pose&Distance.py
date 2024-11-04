import cv2
import numpy as np
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import threading
import time

class KalmanFilter3D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(6, dtype=np.float32)
        np.fill_diagonal(self.kalman.processNoiseCov, 1e-2)
        np.fill_diagonal(self.kalman.measurementNoiseCov, 1e-1)
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)
        self.kalman.statePost = np.zeros(6, dtype=np.float32)

    def update(self, measurement):
        self.kalman.correct(np.array(measurement, dtype=np.float32))
        prediction = self.kalman.predict()
        return prediction[:3].flatten()

class FrameProcessor(threading.Thread):
    def __init__(self, o3r, fg_rgb, fg_dist, camera_matrix, distortion_coeffs, marker_length):
        super().__init__()
        self.o3r = o3r
        self.fg_rgb = fg_rgb
        self.fg_dist = fg_dist
        self.running = True
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.marker_length = marker_length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.kalman_filters = {}

    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)

    def run(self):
        while self.running:
            try:
                ok_rgb, frame_rgb = self.fg_rgb.wait_for_frame().wait_for(150)
                ok_dist, frame_dist = self.fg_dist.wait_for_frame().wait_for(150)

                if ok_rgb and ok_dist:
                    rgb = frame_rgb.get_buffer(buffer_id.JPEG_IMAGE)
                    nparr_rgb = np.frombuffer(rgb, np.uint8)
                    image = cv2.imdecode(nparr_rgb, cv2.IMREAD_COLOR)
                    image = self.undistort_image(image)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

                    if ids is not None:
                        ids = ids.flatten()  # Simplify access to marker IDs
                        cv2.aruco.drawDetectedMarkers(image, corners, ids)
                        marker_positions = {}

                        for i, corner in enumerate(corners):
                            retval, rvec, tvec = cv2.solvePnP(
                                np.array([[0, 0, 0],
                                          [self.marker_length, 0, 0],
                                          [self.marker_length, self.marker_length, 0],
                                          [0, self.marker_length, 0]], dtype=np.float32),
                                corner,
                                self.camera_matrix,
                                self.distortion_coeffs
                            )
                            if retval:
                                marker_id = ids[i]
                                if marker_id not in self.kalman_filters:
                                    self.kalman_filters[marker_id] = KalmanFilter3D()

                                smoothed_position = self.kalman_filters[marker_id].update(tvec.flatten())
                                marker_positions[marker_id] = smoothed_position
                                distance_to_camera = np.linalg.norm(smoothed_position)

                                center_x, center_y = map(int, corner[0][0][:2])
                                cv2.putText(image, f"Marker {marker_id}: ({smoothed_position[0]:.2f}, {smoothed_position[1]:.2f}, {smoothed_position[2]:.2f}), Dist: {distance_to_camera:.2f}",
                                            (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if len(marker_positions) >= 2:
                            point1 = marker_positions[ids[0]]
                            point2 = marker_positions[ids[1]]
                            horizontal_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                            depth_distance = abs(point2[2] - point1[2])

                            print(f"Smoothed Horizontal distance between Marker {ids[0]} and Marker {ids[1]}: {horizontal_distance:.2f} units")
                            print(f"Smoothed Relative depth distance between Marker {ids[0]} and Marker {ids[1]}: {depth_distance:.2f} units")

                    cv2.imshow('RGB Frame', image)
                    cv2.waitKey(1)
                else:
                    print("Frame not available.")
            except Exception as e:
                print(f"Error processing frames: {str(e)}")

    def stop(self):
        self.running = False

def main():
    camera_matrix = np.array([[572.18479771, 0, 649.47702956],
                              [0, 575.19811015, 365.62567089],
                              [0, 0, 1]])
    distortion_coeffs = np.array([-0.2823782, 0.08678044, 0.00332568, -0.00053831, -0.01242341])

    o3r = O3R("192.168.21.73")
    fg_rgb = FrameGrabber(o3r, pcic_port=50013)
    fg_dist = FrameGrabber(o3r, pcic_port=50011)

    fg_rgb.start([buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.JPEG_IMAGE])
    fg_dist.start([buffer_id.RADIAL_DISTANCE_IMAGE])

    processor = FrameProcessor(o3r, fg_rgb, fg_dist, camera_matrix, distortion_coeffs, marker_length=0.05)
    processor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        processor.stop()
        processor.join()
        fg_rgb.stop()
        fg_dist.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
