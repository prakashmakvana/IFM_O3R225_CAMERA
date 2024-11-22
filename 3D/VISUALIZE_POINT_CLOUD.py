import open3d as o3d
import os

# Path to the folder containing the saved point clouds
point_cloud_folder = "captured_point_clouds_50010"

def visualize_point_cloud(file_path):
    try:
        # Load the specified point cloud
        print(f"Loading point cloud: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)

        # Visualize the point cloud
        print("Visualizing point cloud. Use mouse/keyboard to navigate.")
        o3d.visualization.draw_geometries([pcd])

    except Exception as e:
        print(f"Error loading or visualizing point cloud: {e}")

def main():
    # List all .ply files in the folder
    ply_files = [f for f in os.listdir(point_cloud_folder) if f.endswith('.ply')]
    ply_files.sort()  # Sort files alphabetically or by timestamp

    if not ply_files:
        print("No point cloud files found in the folder.")
        return

    print("Available point clouds:")
    for i, file_name in enumerate(ply_files):
        print(f"{i + 1}: {file_name}")

    # Ask user to select a file by name
    file_name = input("Enter the file name of the point cloud you want to visualize: ").strip()

    # Construct the full file path
    file_path = os.path.join(point_cloud_folder, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File '{file_name}' not found in folder '{point_cloud_folder}'.")
        return

    # Visualize the selected point cloud
    visualize_point_cloud(file_path)

if __name__ == "__main__":
    main()
