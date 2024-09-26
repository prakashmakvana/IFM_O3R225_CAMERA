import open3d as o3d
import numpy as np
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import time
import threading

class PointCloudCapture:
    def __init__(self, o3r_ip, pcic_port):
        self.o3r_ip = o3r_ip
        self.pcic_port = pcic_port
        self.pcd = o3d.geometry.PointCloud()
        self.fg = FrameGrabber(O3R(o3r_ip), pcic_port=pcic_port)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start_capture(self):
        self.fg.start([buffer_id.XYZ])
        self.fg.on_new_frame(self._callback)
        print(f"Capturing from port {self.pcic_port}")

    def stop_capture(self):
        self.fg.stop()

    def _callback(self, frame):
        try:
            xyz = frame.get_buffer(buffer_id.XYZ)
            if xyz is None:
                print(f"No XYZ data received from port {self.pcic_port}")
                return

            xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
            if xyz.size == 0:
                print(f"Empty XYZ data from port {self.pcic_port}")
                return

            with self._lock:
                self.pcd.points = o3d.utility.Vector3dVector(xyz)

        except Exception as e:
            print(f"Error in callback from port {self.pcic_port}: {e}")

    def get_point_cloud(self):
        with self._lock:
            return self.pcd

    def stop_event(self):
        self._stop_event.set()

def create_coordinate_frame(size=0.1):
    """ Create a coordinate frame manually. """
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=size))
    return line_set

def filter_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0):
    """ Apply noise filtering to the point cloud without downsampling. """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    return pcd

def register_point_clouds(pcd1, pcd2, max_correspondence_distance=0.05):
    """ Perform point cloud registration using ICP. """
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    icp_result = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, max_correspondence_distance, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp_result

def main():
    print("Initializing point cloud capture")

    capture1 = PointCloudCapture("192.168.21.73", 50010)
    capture2 = PointCloudCapture("192.168.21.73", 50011)

    capture1.start_capture()
    capture2.start_capture()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    rotation_180_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_180_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    try:
        frame_count = 0
        while True:
            pcd1 = capture1.get_point_cloud()
            pcd2 = capture2.get_point_cloud()

            if np.asarray(pcd1.points).shape[0] == 0 or np.asarray(pcd2.points).shape[0] == 0:
                time.sleep(0.05)  # Reduced delay
                continue

            pcd1 = filter_point_cloud(pcd1)
            pcd2 = filter_point_cloud(pcd2)

            icp_result = register_point_clouds(pcd1, pcd2)
            pcd2.transform(icp_result.transformation)
            pcd_combined = pcd1 + pcd2

            pcd_combined.rotate(rotation_180_x, center=(0, 0, 0))
            pcd_combined.rotate(rotation_180_y, center=(0, 0, 0))

            if frame_count % 1 == 0:  # Update visualization every 3 frames
                vis.clear_geometries()
                vis.add_geometry(pcd_combined)
                vis.add_geometry(create_coordinate_frame(size=0.1))
                vis.poll_events()
                vis.update_renderer()

            frame_count += 1
            time.sleep(0.02)  # Smaller delay

    except KeyboardInterrupt:
        print("Stopping capture")
    finally:
        capture1.stop_event()
        capture2.stop_event()
        capture1.stop_capture()
        capture2.stop_capture()
        vis.destroy_window()

if __name__ == "__main__":
    main()
