import open3d as o3d
import numpy as np

def load_and_inspect_ply(ply_path):
    """
    Load and inspect a PLY file
    
    Args:
        ply_path (str): Path to PLY file
        
    Returns:
        o3d.geometry.PointCloud: Loaded point cloud
    """
    try:
        # Load the point cloud
        print(f"\nLoading PLY file: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if not pcd:
            raise RuntimeError(f"Failed to load PLY file: {ply_path}")
            
        # Get basic information
        print("\nPoint Cloud Information:")
        print(f"Number of points: {len(pcd.points)}")
        print(f"Has colors: {pcd.has_colors()}")
        print(f"Has normals: {pcd.has_normals()}")
        
        # Get geometric information
        points = np.asarray(pcd.points)
        print("\nGeometric Information:")
        print(f"Center: {points.mean(axis=0)}")
        print(f"Min bound: {points.min(axis=0)}")
        print(f"Max bound: {points.max(axis=0)}")
        
        # Calculate and print dimensions
        dimensions = points.max(axis=0) - points.min(axis=0)
        print(f"\nDimensions (meters):")
        print(f"Width (X): {dimensions[0]:.3f}")
        print(f"Height (Y): {dimensions[1]:.3f}")
        print(f"Depth (Z): {dimensions[2]:.3f}")
        
        return pcd
        
    except Exception as e:
        print(f"Error inspecting PLY file: {str(e)}")
        return None

def visualize_point_cloud(pcd, show_coordinate_frame=True):
    """
    Visualize the point cloud with optional features
    
    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to visualize
        show_coordinate_frame (bool): Whether to show coordinate frame
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # Add coordinate frame if requested
    if show_coordinate_frame:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5,  # Size of the coordinate frame
            origin=[0, 0, 0]  # Origin of the coordinate frame
        )
        vis.add_geometry(coordinate_frame)
    
    # Optimize view
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def analyze_point_cloud_density(pcd, voxel_size=0.1):
    """
    Analyze point cloud density using voxel downsampling
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud
        voxel_size (float): Size of voxel for density analysis
        
    Returns:
        tuple: (points_per_cubic_meter, average_nearest_neighbor_distance)
    """
    # Calculate volume
    points = np.asarray(pcd.points)
    dimensions = points.max(axis=0) - points.min(axis=0)
    volume = np.prod(dimensions)
    
    # Calculate points per cubic meter
    points_per_cubic_meter = len(pcd.points) / volume
    
    # Calculate average nearest neighbor distance
    if len(pcd.points) > 1:
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        distances = []
        for i in range(len(pcd.points)):
            [_, idx, dist] = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)  # 2 because closest point is self
            if len(dist) > 1:
                distances.append(np.sqrt(dist[1]))  # Get distance to nearest neighbor
        avg_nearest_neighbor = np.mean(distances) if distances else 0
    else:
        avg_nearest_neighbor = 0
    
    print(f"\nDensity Analysis:")
    print(f"Points per cubic meter: {points_per_cubic_meter:.2f}")
    print(f"Average nearest neighbor distance: {avg_nearest_neighbor:.3f} meters")
    
    return points_per_cubic_meter, avg_nearest_neighbor

def main():
    ply_path = "combined_point_cloud.ply"  # Update this path if needed
    
    # Load and inspect the PLY file
    pcd = load_and_inspect_ply(ply_path)
    
    if pcd is not None:
        # Analyze point cloud density
        analyze_point_cloud_density(pcd)
        
        # Visualize the point cloud
        print("\nOpening visualizer...")
        print("Controls:")
        print("- Left click + drag: Rotate")
        print("- Right click + drag: Pan")
        print("- Mouse wheel: Zoom")
        print("- 'H': Show help message")
        print("- 'Q' or 'ESC': Close visualizer")
        
        visualize_point_cloud(pcd)

if __name__ == "__main__":
    main()