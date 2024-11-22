import open3d as o3d
import numpy as np

# File paths for the point clouds
file1 = "captured_point_clouds_50010/point_cloud_20241120_145228.ply"
file2 = "captured_point_clouds_50011/point_cloud_20241120_145326.ply"

def load_point_cloud(file_path):
    print(f"Loading point cloud: {file_path}")
    return o3d.io.read_point_cloud(file_path)

def preprocess_point_cloud(pcd, voxel_size):
    print("Downsampling point cloud...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print("Estimating normals...")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd_down

def compute_fpfh_features(pcd, voxel_size):
    print("Computing FPFH features...")
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    print("Performing global registration with RANSAC...")
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result.transformation

def refine_registration(source, target, initial_transformation, voxel_size):
    print("Refining registration using ICP...")
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result.transformation

def stitch_point_clouds(file1, file2):
    # Load the point clouds
    source = load_point_cloud(file1)
    target = load_point_cloud(file2)

    # Set the voxel size for preprocessing
    voxel_size = 0.05  # Adjust based on your point cloud scale

    # Preprocess point clouds
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    # Compute FPFH features
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)

    # Perform global registration
    transformation_global = global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("Global transformation:")
    print(transformation_global)

    # Perform fine registration
    transformation_icp = refine_registration(source, target, transformation_global, voxel_size)
    print("Refined transformation:")
    print(transformation_icp)

    # Apply the transformation to align source with target
    source.transform(transformation_icp)

    # Merge the point clouds
    print("Combining point clouds...")
    stitched = source + target
    stitched_down = stitched.voxel_down_sample(voxel_size)  # Downsample to reduce density

    # Save the result
    output_file = "stitched_point_cloud.ply"
    o3d.io.write_point_cloud(output_file, stitched_down)
    print(f"Stitched point cloud saved: {output_file}")

    # Visualize the result
    o3d.visualization.draw_geometries([stitched_down], window_name="Stitched Point Cloud")

if __name__ == "__main__":
    stitch_point_clouds(file1, file2)
