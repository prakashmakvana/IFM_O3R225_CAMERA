import open3d as o3d
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import numpy as np
import time

# Global variables for the Open3D visualizer and point clouds
vis = None
pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
geometry_added = False

def callback1(frame):
    global pcd1
    update_point_cloud(frame, pcd1, "Camera 1")

def callback2(frame):
    global pcd2
    update_point_cloud(frame, pcd2, "Camera 2")

def update_point_cloud(frame, pcd, camera_name):
    global geometry_added
    print(f"{camera_name} Callback triggered")
    try:
        xyz = frame.get_buffer(buffer_id.XYZ)
        if xyz is None:
            print(f"No XYZ data received from {camera_name}")
            return
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if not geometry_added:
            vis.add_geometry(pcd)
            geometry_added = True
    except Exception as e:
        print(f"Error in {camera_name} callback: {e}")

def visualize():
    global vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    
    while True:
        vis.update_geometry(pcd1)
        vis.update_geometry(pcd2)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()

def compute_transformation(pcd1, pcd2):
    print("Computing transformation between point clouds")
    # Use ICP to find the best alignment
    threshold = 0.02  # Maximum distance threshold for ICP
    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # Print ICP registration results
    print("ICP Fitness:", reg_icp.fitness)
    print("ICP Inlier RMSE:", reg_icp.inlier_rmse)
    print("Transformation matrix:\n", reg_icp.transformation)
    
    return reg_icp.transformation

def stitch_point_clouds(pcd1, pcd2):
    # Compute transformation from pcd2 to pcd1
    transformation = compute_transformation(pcd1, pcd2)
    print("Applying transformation to pcd2")
    pcd2.transform(transformation)
    pcd_combined = pcd1 + pcd2
    return pcd_combined

def main():
    print("Initializing O3R and FrameGrabber for Camera 1")
    o3r1 = O3R("192.168.21.73")
    fg1 = FrameGrabber(o3r1, pcic_port=50010)
    fg1.start([buffer_id.XYZ])
    fg1.on_new_frame(callback1)

    print("Initializing O3R and FrameGrabber for Camera 2")
    o3r2 = O3R("192.168.21.73")
    fg2 = FrameGrabber(o3r2, pcic_port=50011)
    fg2.start([buffer_id.XYZ])
    fg2.on_new_frame(callback2)

    try:
        # Start visualization in the main thread
        visualize()

    except KeyboardInterrupt:
        pass

    # Stop FrameGrabbers
    fg1.stop()
    fg2.stop()
    print("FrameGrabbers stopped")

    # Stitch and visualize the combined point cloud
    stitched_pcd = stitch_point_clouds(pcd1, pcd2)
    
    # Visualize the individual and stitched point clouds
    print("Visualizing individual point clouds before transformation")
    o3d.visualization.draw_geometries([pcd1], window_name="Camera 1 Point Cloud")
    o3d.visualization.draw_geometries([pcd2], window_name="Camera 2 Point Cloud")
    
    print("Visualizing transformed and combined point cloud")
    o3d.visualization.draw_geometries([stitched_pcd], window_name="Stitched Point Cloud")

if __name__ == "__main__":
    main()
