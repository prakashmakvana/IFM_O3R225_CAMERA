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

def capture_different_angles(processor, num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    num = 0
    start_time = time.time()
    capture_interval = 10  # Interval between captures in seconds

    # Define the view angles and their corresponding directions
    directions = ["front", "front_left", "front_right", "back", "back_left", "back_right"]

    while num < num_images:
        frame = processor.get_frame()
        if frame is not None:
            cv2.imshow('Calibration', frame)
            print("Frame displayed.")
            print(f"Frame dimensions: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")

            current_time = time.time()
            if current_time - start_time >= capture_interval:
                # Resize the frame to 224x224
                resized_frame = cv2.resize(frame, (224, 224))

                # Use direction to create unique filenames
                direction = directions[num % len(directions)]
                filename = os.path.join(save_dir, f'{direction}_{num}.png')
                print(f"Saving image as {filename}")
                try:
                    success = cv2.imwrite(filename, resized_frame)
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
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        save_dir = os.path.join(desktop_path, "captured_images")
        num_images = 6  # Number of images to capture (one for each angle)

        # Start frame processor
        processor.start()
        print("Frame processor started.")

        # Capture images from different angles
        capture_different_angles(processor, num_images, save_dir)

        # Stop processor and frame grabber
        processor.stop()
        fg.stop()
        processor.join()

    except Exception as e:
        print(f"Exception in main: {e}")

if __name__ == "__main__":
    main()
