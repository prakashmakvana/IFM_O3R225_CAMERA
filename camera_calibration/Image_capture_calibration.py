import cv2
import numpy as np
import os
import time
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import threading

class FrameProcessor(threading.Thread):
    def __init__(self, o3r, fg):
        super(FrameProcessor, self).__init__()
        self.o3r = o3r
        self.fg = fg
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            try:
                [ok, frame] = self.fg.wait_for_frame().wait_for(150)
                if ok:
                    rgb = frame.get_buffer(buffer_id.JPEG_IMAGE)
                    nparr = np.frombuffer(rgb, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is not None:
                        with self.lock:
                            self.frame = image
                else:
                    print("Failed to receive frame")
            except Exception as e:
                print(f"Exception in frame processing: {e}")
            time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False

def capture_chessboard_images(processor, num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    num = 0
    start_time = time.time()
    capture_interval = 5  # Interval between captures in seconds

    while num < num_images:
        frame = processor.get_frame()
        if frame is not None:
            cv2.imshow('Calibration', frame)
            print("Frame displayed.")
            print(f"Frame dimensions: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")

            current_time = time.time()
            if current_time - start_time >= capture_interval:
                filename = os.path.join(save_dir, f'img_{num}.png')
                print(f"Saving image as {filename}")
                try:
                    success = cv2.imwrite(filename, frame)
                    if success:
                        print(f"Image saved as {filename}")
                        num += 1
                    else:
                        print(f"Failed to save image as {filename}")
                except Exception as e:
                    print(f"Exception occurred while saving image: {e}")

                start_time = current_time

        k = cv2.waitKey(1)
        if k == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()

def calibrate_camera(images_dir, chessboard_size):
    # Prepare object points and image points
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    # Read images
    images = [os.path.join(images_dir, fname) for fname in sorted(os.listdir(images_dir)) if fname.endswith('.png')]

    for image_file in images:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)
        else:
            print(f"Chessboard corners not found in {image_file}")

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("Calibration successful.")
        print("Camera matrix:\n", K)
        print("Distortion coefficients:\n", dist)
        return K, dist
    else:
        print("Calibration failed.")
        return None, None

def main():
    try:
        # Create O3R device instance
        o3r = O3R("192.168.21.73")
        print("Connected to O3R device.")

        # Create frame grabber for port 50012
        fg = FrameGrabber(o3r, pcic_port=50013)
        fg.start([buffer_id.JPEG_IMAGE])
        print("Frame grabber started on port 50013.")

        # Create frame processor
        processor = FrameProcessor(o3r, fg)

        # Create the directory to save calibration images if it doesn't exist
        calibration_images_dir = "/home/demo/Desktop/calibration_50013"
        num_images = 20  # Number of images to capture

        # Start frame processor
        processor.start()
        print("Frame processor started.")

        # Capture images
        capture_chessboard_images(processor, num_images, calibration_images_dir)

        # Stop processor and frame grabber
        processor.stop()
        fg.stop()
        processor.join()

        # Perform camera calibration
        chessboard_size = (8, 6)  # Number of inner corners per chessboard row and column
        K, dist = calibrate_camera(calibration_images_dir, chessboard_size)

        if K is not None and dist is not None:
            # Save calibration data
            np.savez(os.path.join(calibration_images_dir, 'calibration_data.npz'), camera_matrix=K, distortion_coefficients=dist)
            print("Calibration data saved.")
        else:
            print("Calibration data not saved.")

    except Exception as e:
        print(f"Exception in main: {e}")

if __name__ == "__main__":
    main()
