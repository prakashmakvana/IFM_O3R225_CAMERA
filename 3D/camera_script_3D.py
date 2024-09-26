import open3d as o3d
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import numpy as np
import time

# Global variables for the Open3D visualizer and point cloud
vis = None
pcd = o3d.geometry.PointCloud()
geometry_added = False

def callback(frame):
    global pcd, geometry_added

    print("Callback triggered")

    try:
        # Read the XYZ data
        xyz = frame.get_buffer(buffer_id.XYZ)

        # Check if data is being received
        if xyz is None:
            print("No XYZ data received")
            return

        # Convert to numpy array if necessary and reshape
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

        # Check the shape of the data
        print(f"XYZ data shape: {xyz.shape}")

        # Update Open3D point cloud object
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Add the geometry only once
        if not geometry_added:
            vis.add_geometry(pcd)
            geometry_added = True

    except Exception as e:
        print(f"Error in callback: {e}")

def visualize():
    global vis, pcd

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add a dummy point to initialize the point cloud
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    while True:
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)  # Adjust this for your refresh rate

    vis.destroy_window()

def main():
    print("Initializing O3R and FrameGrabber")
    # Initialize the objects
    o3r = O3R("192.168.21.73")  # Assuming default parameters
    fg = FrameGrabber(o3r, pcic_port=50011)

    # Set schema and start Grabber with XYZ buffer only
    fg.start([buffer_id.XYZ])
    print("FrameGrabber started")

    # Register callback for new frames
    fg.on_new_frame(callback)
    print("Callback registered")

    try:
        # Start the visualization in the main thread
        visualize()

    except KeyboardInterrupt:
        pass

    # Stop FrameGrabber when done
    fg.stop()
    print("FrameGrabber stopped")

if __name__ == "__main__":
    main()
