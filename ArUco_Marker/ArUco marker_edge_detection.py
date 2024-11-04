import cv2
import numpy as np
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import threading
import time

class FrameProcessor(threading.Thread):
    def __init__(self, o3r, fg_rgb, fg_dist, camera_matrix, distortion_coeffs):
        super(FrameProcessor, self).__init__()
        self.o3r = o3r
        self.fg_rgb = fg_rgb
        self.fg_dist = fg_dist
        self.running = True
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs

        # Set up the ArUco parameters and dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.last_coordinates = {}

    def run(self):
        while self.running:
            try:
                # Wait for RGB frame
                [ok_rgb, frame_rgb] = self.fg_rgb.wait_for_frame().wait_for(150)  # wait with 150ms timeout
                # Wait for distance frame
                [ok_dist, frame_dist] = self.fg_dist.wait_for_frame().wait_for(150)  # wait with 150ms timeout

                if ok_rgb and ok_dist:
                    # Process the RGB frame
                    rgb = frame_rgb.get_buffer(buffer_id.JPEG_IMAGE)
                    nparr_rgb = np.frombuffer(rgb, np.uint8)
                    image = cv2.imdecode(nparr_rgb, cv2.IMREAD_COLOR)

                    # Calculate the optical center
                    optical_center_x = int(self.camera_matrix[0, 2])  # cx
                    optical_center_y = int(self.camera_matrix[1, 2])  # cy

                    # Draw a circle at the optical center
                    cv2.circle(image, (optical_center_x, optical_center_y), 5, (0, 255, 0), -1)  # Green circle

                    # Process the distance frame
                    dist = frame_dist.get_buffer(buffer_id.RADIAL_DISTANCE_IMAGE)
                    
                    if dist is None:
                        print("Distance frame is None.")
                        continue

                    # Get dimensions of the RGB and distance images
                    rgb_height, rgb_width = image.shape[:2]
                    dist_height, dist_width = dist.shape

                    print(f"RGB image dimensions: {rgb_width} x {rgb_height}")
                    print(f"Distance image dimensions: {dist_width} x {dist_height}")

                    # Detect ArUco markers
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

                    # Convert distance image to display format
                    dist_display = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
                    dist_display = np.uint8(dist_display)
                    
                    # Convert distance image to BGR for color edge overlay
                    dist_display_color = cv2.cvtColor(dist_display, cv2.COLOR_GRAY2BGR)

                    # If markers are detected, draw them and calculate 3D coordinates
                    if ids is not None:
                        ids = np.array(ids)  # Ensure ids is a NumPy array
                        cv2.aruco.drawDetectedMarkers(image, corners, ids)

                        for i, corner in enumerate(corners):
                            # Calculate the center of the marker
                            center_x = int((corner[0][0][0] + corner[0][2][0]) / 2)
                            center_y = int((corner[0][0][1] + corner[0][2][1]) / 2)

                            # Scale center coordinates to distance image dimensions
                            scaled_center_x = int(center_x * (dist_width / rgb_width))
                            scaled_center_y = int(center_y * (dist_height / rgb_height))

                            # Define a fixed size for the ROI (e.g., 40x40 pixels)
                            roi_size = 40

                            # Calculate the ROI based on the marker's scaled center
                            x_start = scaled_center_x - roi_size // 2
                            y_start = scaled_center_y - roi_size // 2
                            x_end = x_start + roi_size
                            y_end = y_start + roi_size

                            # Ensure the ROI is within the image bounds
                            x_start = max(0, x_start)
                            y_start = max(0, y_start)
                            x_end = min(dist_width, x_end)
                            y_end = min(dist_height, y_end)

                            # Debugging print statements
                            print(f"Marker ID: {ids[i][0]}, Center: ({center_x}, {center_y}), "
                                  f"Scaled Center: ({scaled_center_x}, {scaled_center_y}), "
                                  f"ROI: ({x_start}, {y_start}, {x_end}, {y_end})")

                            # Draw the center of the ArUco marker in blue
                            cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue circle

                            # Draw the center of the ROI in red
                            roi_center_x = (x_start + x_end) // 2
                            roi_center_y = (y_start + y_end) // 2
                            cv2.circle(dist_display_color, (roi_center_x, roi_center_y), 5, (0, 0, 255), -1)  # Red circle

                            # Extract the ROI from the distance image
                            roi = dist[y_start:y_end, x_start:x_end]

                            # Normalize ROI for edge detection
                            roi_normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
                            roi_normalized = np.uint8(roi_normalized)

                            # Apply Gaussian Blur to reduce noise
                            roi_blurred = cv2.GaussianBlur(roi_normalized, (5, 5), 0)

                            # Apply Canny Edge Detection
                            edges = cv2.Canny(roi_blurred, threshold1=50, threshold2=150)

                            # Overlay the edges on the distance_display_color image in red color
                            dist_display_color[y_start:y_end, x_start:x_end][edges != 0] = [0, 0, 255]  # Red edges

                            # Optionally, draw a rectangle around the ROI
                            cv2.rectangle(dist_display_color, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)  # Green rectangle

                            # Get the distance at the scaled center of the marker
                            if 0 <= scaled_center_x < dist_width and 0 <= scaled_center_y < dist_height:
                                z = dist[scaled_center_y, scaled_center_x]  # Distance from the camera
                                # Compute the 3D coordinates
                                x = (scaled_center_x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                                y = (scaled_center_y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]

                                # Only print if coordinates have changed
                                if ids[i][0] not in self.last_coordinates or self.last_coordinates[ids[i][0]] != (x, y, z):
                                    print(f"Marker ID: {ids[i][0]}, 3D Coordinates: ({x:.2f}, {y:.2f}, {z:.2f}), Distance: {z:.2f} units")
                                    self.last_coordinates[ids[i][0]] = (x, y, z)
                            else:
                                print(f"Marker ID: {ids[i][0]} is out of bounds for distance image.")
                    else:
                        print("No markers detected.")

                    # Display the images using OpenCV
                    cv2.imshow('RGB Frame', image)
                    cv2.imshow('Distance Image', dist_display_color)  # Display the color distance image with edges

                    cv2.waitKey(1)  # Adjust waitKey value for frame rate
                else:
                    print("Frame not available.")

            except Exception as e:
                print(f"Error processing frames: {str(e)}")

    def stop(self):
        self.running = False

def main():
    # Camera calibration parameters
    camera_matrix = np.array([[593.4302352, 0, 632.52853471],
                              [0, 591.24810738, 418.0353015],
                              [0, 0, 1]])
    distortion_coeffs = np.array([[-0.44504276,  0.6419505,  -0.00134617,  0.00294749, -0.7190250]])

    o3r = O3R("192.168.21.73")
    fg_rgb = FrameGrabber(o3r, pcic_port=50013)  # RGB port
    fg_dist = FrameGrabber(o3r, pcic_port=50011)  # 3D port

    fg_rgb.start([buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.JPEG_IMAGE])
    fg_dist.start([buffer_id.RADIAL_DISTANCE_IMAGE])

    processor = FrameProcessor(o3r, fg_rgb, fg_dist, camera_matrix, distortion_coeffs)
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
        o3r.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
