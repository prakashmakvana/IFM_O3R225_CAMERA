import cv2
import numpy as np
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import threading
import time

class FrameProcessor(threading.Thread):
    def __init__(self, o3r, fg, window_name):
        super(FrameProcessor, self).__init__()
        self.o3r = o3r
        self.fg1 = fg[0]
        self.fg2 = fg[1]
        self.window_name = window_name
        self.running = True

    def run(self):
        while self.running:
            try:
                [ok1, frame1] = self.fg1.wait_for_frame().wait_for(150)  # wait with 150ms timeout
                if ok1:
                    rgb1 = frame1.get_buffer(buffer_id.JPEG_IMAGE)
                    nparr1 = np.frombuffer(rgb1, np.uint8)
                    image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
                else:
                    print(f"Failed to receive frame for {self.window_name}")
            except Exception as e:
                print(f"Exception in frame1: {e}")
            try:
                [ok2, frame2] = self.fg2.wait_for_frame().wait_for(150)  # wait with 150ms timeout
                if ok2:
                    rgb2 = frame2.get_buffer(buffer_id.JPEG_IMAGE)
                    nparr2 = np.frombuffer(rgb2, np.uint8)
                    image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
                else:
                    print(f"Failed to receive frame for {self.window_name}")
            except Exception as e:
                print(f"Exception in frame2: {e}")

            image_all = cv2.hconcat([image2, image1]) 
            if image_all is not None:
                # Display the image using OpenCV
                cv2.imshow(self.window_name, image_all)
            else:
                print(f"Failed to concat frames ")

                # Adjust waitKey value for frame rate
            cv2.waitKey(1)  

            #except Exception as e:
            #    print(f"Exception in frame processor for {self.window_name}: {e}")

    def stop(self):
        self.running = False

def main():
    try:
        # Create O3R device instance
        o3r = O3R("192.168.21.73")
        conf = o3r.get()
        print("Port 3: ", o3r.get([f"/ports/port3/data/pcicTCPPort"]))
        print("Port 2: ", o3r.get([f"/ports/port2/data/pcicTCPPort"]))

        # Create frame grabbers
        fg1 = FrameGrabber(o3r, pcic_port=50013)
        fg2 = FrameGrabber(o3r, pcic_port=50012)
        fg1.start([buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.JPEG_IMAGE])
        fg2.start([buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.JPEG_IMAGE])

        # Create frame processors
        processor1 = FrameProcessor(o3r, [fg1,fg2], 'Frame1')
        #processor2 = FrameProcessor2(o3r, fg2)

        # Start frame processors
        processor1.start()
        #processor2.start()

        try:
            while True:
                time.sleep(1)  # Main thread can do other tasks if needed

        except KeyboardInterrupt:
            print("Stopping...")

        finally:
            # Stop processors and frame grabbers
            processor1.stop()
            #processor2.stop()
            fg1.stop()
            fg2.stop()

            # Wait for threads to finish
            processor1.join()
            #processor2.join()

            # Close OpenCV windows
            cv2.destroyAllWindows()
            print("Stopped all processors and closed OpenCV windows")

    except Exception as e:
        print(f"Exception in main: {e}")

if __name__ == "__main__":
    main()
