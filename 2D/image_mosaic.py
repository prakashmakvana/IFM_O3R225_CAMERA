import sys
import cv2
import numpy as np
import threading
import time
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
from sift_matching import SiftMatching  # Ensure this file is in the same directory or adjust the import

class FrameProcessor(threading.Thread):
    def __init__(self, fg1, fg2, window_name):
        super(FrameProcessor, self).__init__()
        self.fg1 = fg1
        self.fg2 = fg2
        self.window_name = window_name
        self.running = True
        self.detector = cv2.SIFT_create()  # Using SIFT for feature detection
        # FLANN parameters for SIFT
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def run(self):
        while self.running:
            try:
                # Capture frames
                image1, image2 = self.capture_frames()
                if image1 is not None and image2 is not None:
                    # Perform image stitching
                    stitched_image = self.stitch_images(image1, image2)
                    # Display the stitched image
                    if stitched_image is not None:
                        cv2.imshow(self.window_name, stitched_image)
                    else:
                        print("Image stitching failed.")
                else:
                    print("One or both images are None")

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Adjust to control frame rate and exit condition
                    break

            except Exception as e:
                print(f"Exception in frame processor: {e}")

        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed when stopping

    def capture_frames(self):
        try:
            # Capture frame from the first camera
            ok1, frame1 = self.fg1.wait_for_frame().wait_for(150)
            if ok1:
                rgb1 = frame1.get_buffer(buffer_id.JPEG_IMAGE)
                nparr1 = np.frombuffer(rgb1, np.uint8)
                image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
            else:
                image1 = None
                print("Failed to receive frame from first camera")

            # Capture frame from the second camera
            ok2, frame2 = self.fg2.wait_for_frame().wait_for(150)
            if ok2:
                rgb2 = frame2.get_buffer(buffer_id.JPEG_IMAGE)
                nparr2 = np.frombuffer(rgb2, np.uint8)
                image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
            else:
                image2 = None
                print("Failed to receive frame from second camera")

            return image1, image2

        except Exception as e:
            print(f"Exception in capturing frames: {e}")
            return None, None

    def stitch_images(self, img1, img2):
        # Use the SiftMatching class for feature extraction and matching
        sift_match = SiftMatching(img1, img2, nfeatures=2000, gamma=0.8)
        correspondence = sift_match.run()

        if len(correspondence) >= 10:
            src_pts = np.float32([c[:2] for c in correspondence]).reshape(-1, 2)
            dst_pts = np.float32([c[2:] for c in correspondence]).reshape(-1, 2)

            # Use RANSAC to estimate the homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Warp the first image to the perspective of the second image
                height, width, _ = img2.shape
                panorama = cv2.warpPerspective(img1, H, (width + img1.shape[1], height))

                # Blend the images
                panorama[0:img2.shape[0], 0:img2.shape[1]] = img2

                # Visualize matches
                self.visualize_matches(img1, src_pts, img2, dst_pts, mask)

                return panorama
            else:
                print("Homography computation failed")
                return None
        else:
            print(f"Not enough good matches found - {len(correspondence)}/10")
            return None

    def visualize_matches(self, img1, src_pts, img2, dst_pts, mask):
        try:
            inlier_matches = np.hstack((src_pts[mask.ravel() == 1], dst_pts[mask.ravel() == 1]))
            img_matches = self.draw_correspondence(inlier_matches, img1, img2)
            cv2.imshow('Matches', img_matches)
            cv2.waitKey(1)  # Display the matches for a brief moment
        except Exception as e:
            print(f"Error visualizing matches: {e}")

    def draw_correspondence(self, correspondence, img1, img2):
        h, w, _ = img1.shape
        img_stack = np.hstack((img1, img2))
        for x1, y1, x2, y2 in correspondence:
            x1_d = int(round(x1))
            y1_d = int(round(y1))
            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))
            cv2.circle(img_stack, (x1_d, y1_d), radius=5, color=[255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img_stack, (x2_d, y2_d), radius=5, color=[255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img_stack, (x1_d, y1_d), (x2_d, y2_d), color=[255, 255, 0], thickness=2)
        return img_stack

    def stop(self):
        self.running = False

def main():
    try:
        # Create O3R device instance
        o3r = O3R("192.168.21.73")
        print("Port 3: ", o3r.get([f"/ports/port3/data/pcicTCPPort"]))
        print("Port 2: ", o3r.get([f"/ports/port2/data/pcicTCPPort"]))

        # Create frame grabbers
        fg1 = FrameGrabber(o3r, pcic_port=50013)
        fg2 = FrameGrabber(o3r, pcic_port=50012)
        fg1.start([buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.JPEG_IMAGE])
        fg2.start([buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.JPEG_IMAGE])

        # Create and start frame processor
        processor = FrameProcessor(fg1, fg2, 'Panorama')
        processor.start()

        try:
            while True:
                time.sleep(1)  # Main thread can do other tasks if needed

        except KeyboardInterrupt:
            print("Stopping...")

        finally:
            # Stop processor and frame grabbers
            processor.stop()
            fg1.stop()
            fg2.stop()

            # Wait for thread to finish
            processor.join()

            # Close OpenCV windows
            cv2.destroyAllWindows()
            print("Stopped all processors and closed OpenCV windows")

    except Exception as e:
        print(f"Exception in main: {e}")

if __name__ == "__main__":
    main()
