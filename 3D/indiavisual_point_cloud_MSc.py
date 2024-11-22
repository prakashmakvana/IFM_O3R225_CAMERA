import open3d as o3d
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id
import numpy as np
import time
import asyncio

import IFM_CAM_RUN
import IFM_CAM_IDLE

import os
from pathlib import Path
import json

# Global variables for the Open3D visualizer and point clouds
vis = None
vis2 = None
pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
stitched_pcd = o3d.geometry.PointCloud()
transformation = []
geometry_added = False

def callback1(frame):
    global pcd1
    update_point_cloud(frame, pcd1, "Camera 1")

def callback2(frame):
    global pcd2
    update_point_cloud(frame, pcd2, "Camera 2")

def update_point_cloud(frame, pcd, camera_name):
    global geometry_added
    global transformation
    #print(f"{camera_name} Callback triggered")
    try:
        xyz = frame.get_buffer(buffer_id.XYZ)
        if xyz is None:
            print(f"No XYZ data received from {camera_name}")
            return
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        rotation_matrix = np.array([[1.0,  0.0,  0.0,  0.0],
                                  [0.0, -1.0,  0.0,  0.0],
                                  [0.0,  0.0,  -1.0,  0.0],
                                  [0.0,  0.0,  0.0,  1.0]])
        pcd.transform(rotation_matrix)
        if camera_name == "Camera 2":
            tranz = True
            if tranz:
                grad = -14 * 2 * np.pi / 360.0
                tx = 0.6 #-2.5 #-1.35
                ty = -0.02
                tz = 0.05
                transformation_y = np.array([[np.cos(grad),  0.0,  np.sin(grad),  tx],
                                            [0.0,  1.0,  0.0,  ty],
                                            [-np.sin(grad),  0.0,  np.cos(grad),  tz],
                                            [0.0,  0.0,  0.0,  1.0]])
                #print("transformation_y: ",transformation_y)
                grad = 0.7 * 2 * np.pi / 360.0
                tx = 0.0 #-2.5 #-1.35
                ty = 0.0
                tz = 0.0
                transformation_z = np.array([[np.cos(grad),  -np.sin(grad),  0.0,  tx],
                                            [np.sin(grad),  np.cos(grad),  0.0,  ty],
                                            [0.0,  0.0,  1.0,  tz],
                                            [0.0,  0.0,  0.0,  1.0]])
                
                #print("transformation_z: ",transformation_z)
                transform = np.matmul(transformation_y, transformation_z)
                # transform = np.array([[ 0.96118996, -0.01174373, -0.27563736,  0.4       ],
                #                       [ 0.012217,    0.99992537,  0.,         -0.02      ],
                #                       [ 0.27561678, -0.00336746,  0.9612617,   0.05      ],
                #                       [ 0.,          0.,          0.,          1.        ]])
                pcd.transform(transform)
                #print("transform: ",transform)
            
        #print("type(pcd): ", type(pcd))
        if not geometry_added:
            vis.add_geometry(pcd)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, 
                                                                   origin=[0.0, 0.0, 0.0])
            vis.add_geometry(mesh_frame)
        #    print("\npcd: \n",pcd)
            geometry_added = True
    except Exception as e:
        print(f"Error in {camera_name} callback: {e}")

def visualize():
    global vis#, vis2
    global pcd1, pcd2
    global stitched_pcd
    
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, 
                                                                   origin=[0.0, 0.0, 0.0])
    vis.add_geometry(mesh_frame)
    #stitched_pcd = stitch_point_clouds(pcd1, pcd2)
    #vis2.add_geometry(stitched_pcd)
    #print("type(vis2): ",type(vis2))
    while True:
        
        vis.update_geometry(pcd1)
        #fg_1.wait_for_frame()
        #fg_2.wait_for_frame()
        vis.update_geometry(pcd2)
        vis.poll_events()
        vis.update_renderer()
        # Stitch and visualize the combined point cloud
        #stitched_pcd = stitch_point_clouds(pcd1, pcd2)
        time.sleep(0.5)

    #vis.destroy_window()

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
    global geometry_added
    global stitched_pcd
    global transformation
    #print("\nstitch_point_clouds function!\n")
    #print("pcd1:",pcd1)
    #print((str(pcd1) == "PointCloud with 0 points."))
    #print("pcd2:",pcd2)
    # Compute transformation from pcd2 to pcd1
    if ((str(pcd1) or str(pcd2)) == "PointCloud with 0 points."):
        #print("no pcd data!")
        return 0
    else:
        fname2 = "transformation_save"
        cwd = os.getcwd()
        file_true = os.path.isfile(cwd + os.sep + fname2+".npy")
        if file_true:
            #print("File exists!")
            transformation = np.load(fname2+".npy")
            #print("transformation loaded: ", transformation)

        else:
            #print("File didn't exists!")
            # get all streets from index
            
            transformation = compute_transformation(pcd1, pcd2)
            #print("transformation calculated: ", transformation)
            #print("type(transformation): ",type(transformation))
            
            #print("save transformation in file: ", fname2)
            np.save(fname2, transformation)
            #Path(fname2).write_text(str(transformation))
        return 0

def main():
    global vis
    global stitched_pcd
    print("type(vis): ",type(vis))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    print("type(vis): ",type(vis))
    #print("type(vis2): ",type(vis2))
    #vis2 = o3d.visualization.Visualizer()
    #vis2.create_window()
    #print("type(vis2): ",type(vis2))

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
    
    print("trying to visualize!")
    try:
        # Start visualization in the main thread
        visualize()

    except KeyboardInterrupt:
        # Stop FrameGrabbers
        fg1.stop()
        fg2.stop()
        print("FrameGrabbers stopped")

        # Visualize the individual and stitched point clouds
        print("Visualizing individual point clouds before transformation")
        o3d.visualization.draw_geometries([pcd1], window_name="Camera 1 Point Cloud")
        o3d.visualization.draw_geometries([pcd2], window_name="Camera 2 Point Cloud")
        #o3d.visualization.draw_geometries([stitched_pcd], window_name="Camera 1 + 2 Point Cloud")
        #print("Visualizing transformed and combined point cloud")
        #o3d.visualization.draw_geometries([stitched_pcd], window_name="Stitched Point Cloud")
        vis.destroy_window()
        #vis2.destroy_window()
        exit()

if __name__ == "__main__":
    main()
