import open3d as o3d
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import numpy as np
from sklearn.cluster import DBSCAN
import time

# Global variables
vis = None
pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
geometry_added = False

def callback1(frame):
    global pcd1, geometry_added
    update_point_cloud(frame, pcd1, "Camera 1")

def callback2(frame):
    global pcd2, geometry_added
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
        labels = detect_objects(pcd)  # Cluster detection and coloring
        colors = assign_colors(labels)  # Assign colors based on clusters
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if not geometry_added:
            vis.add_geometry(pcd)
            geometry_added = True
    except Exception as e:
        print(f"Error in {camera_name} callback: {e}")

def detect_objects(pcd):
    # Use DBSCAN to find clusters
    xyz = np.asarray(pcd.points)
    clustering = DBSCAN(eps=0.05, min_samples=10).fit(xyz)
    labels = clustering.labels_
    print(f"Detected clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    return labels

def assign_colors(labels):
    max_label = labels.max()
    colors = np.zeros((labels.size, 3))
    np.random.seed(42)
    for i in range(max_label + 1):
        color = np.random.rand(3)
        colors[labels == i] = color
    colors[labels == -1] = [0, 0, 0]
    return colors

def label_clusters(pcd, labels):
    max_label = labels.max()
    labels_dict = {}
    for i in range(max_label + 1):
        cluster_points = np.asarray(pcd.points)[labels == i]
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cluster_points))
        dimensions = bbox.get_extent()
        if dimensions[2] > 1.0:  # Example threshold for height
            labels_dict[i] = "Board"
        elif dimensions[0] > 0.5 and dimensions[1] > 0.5:
            labels_dict[i] = "Table"
        else:
            labels_dict[i] = "Chair"
    return labels_dict

def visualize_labeled_clusters(pcd, labels, labels_dict):
    for i, label in labels_dict.items():
        cluster_points = np.asarray(pcd.points)[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        vis.add_3d_text(text=label, pos=cluster_center, size=12.0, color=(1, 0, 0))

def visualize():
    global vis, pcd1, pcd2
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    
    while True:
        try:
            vis.update_geometry(pcd1)
            vis.update_geometry(pcd2)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break

    vis.destroy_window()

def stitch_point_clouds(pcd1, pcd2):
    print("Stitching point clouds")
    transformation = np.eye(4)
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

    # Detect and label clusters
    labels = detect_objects(stitched_pcd)
    labels_dict = label_clusters(stitched_pcd, labels)
    visualize_labeled_clusters(stitched_pcd, labels, labels_dict)

    o3d.visualization.draw_geometries([stitched_pcd])

if __name__ == "__main__":
    main()
