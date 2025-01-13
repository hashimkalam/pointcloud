import open3d as o3d
import numpy as np
import os
from PIL import Image

def convert_to_depth(image_np):
    """
    Convert RGB/RGBA image to single-channel depth image
    
    Args:
        image_np (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Single-channel 16-bit depth image
    """
    # If already single channel, just convert to 16-bit
    if len(image_np.shape) == 2:
        depth_np = image_np.astype(np.float32) * (65535.0 / 255.0)
        return depth_np.astype(np.uint16)
    
    # If RGB/RGBA, convert to grayscale first
    if len(image_np.shape) == 3:
        # Convert to grayscale using standard weights
        depth_np = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])
        # Scale to 16-bit range
        depth_np = depth_np.astype(np.float32) * (65535.0 / 255.0)
        return depth_np.astype(np.uint16)
    
    raise ValueError(f"Unsupported image format with shape: {image_np.shape}")

def generate_point_cloud_from_depth(depth_path):
    """
    Generate point cloud from an image
    
    Args:
        depth_path (str): Path to image
    
    Returns:
        o3d.geometry.PointCloud: Generated point cloud
    """
    try:
        # Read image with PIL
        with Image.open(depth_path) as img:
            image_np = np.array(img)
            
        print(f"Original image format: {image_np.dtype}, shape: {image_np.shape}")
        
        # Convert to single-channel depth image
        depth_np = convert_to_depth(image_np)
        print(f"Converted to depth format: {depth_np.dtype}, shape: {depth_np.shape}")
        
        # Save as temporary image for Open3D
        temp_path = depth_path + '_temp.png'
        Image.fromarray(depth_np).save(temp_path)
        depth_image = o3d.io.read_image(temp_path)
        os.remove(temp_path)
            
        if not depth_image:
            raise RuntimeError(f"Failed to load depth image: {depth_path}")
        
        # Create camera intrinsic parameters
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=depth_np.shape[1],
            height=depth_np.shape[0],
            fx=525.0,  # focal length x
            fy=525.0,  # focal length y
            cx=depth_np.shape[1]/2,  # principal point x
            cy=depth_np.shape[0]/2   # principal point y
        )
        
        # Convert depth image to point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            camera_intrinsics,
            depth_scale=1000.0,  # adjust if your depth is in different units
            depth_trunc=3.0,     # maximum depth in meters
        )
        
        # Remove any zero points
        pcd = pcd.remove_non_finite_points()
        
        return pcd
        
    except Exception as e:
        print(f"Error processing {depth_path}: {str(e)}")
        return None

def process_depth_folder(folder_path):
    """
    Process all images in a folder
    
    Args:
        folder_path (str): Path to folder containing images
    
    Returns:
        o3d.geometry.PointCloud: Combined point cloud
    """
    print(f"Processing folder: {folder_path}")
    
    image_files = [f for f in os.listdir(folder_path) 
                  if f.endswith(('.png', '.jpg', '.jpeg')) and 
                  os.path.isfile(os.path.join(folder_path, f))]
    
    if not image_files:
        raise RuntimeError(f"No suitable images found in {folder_path}")
        
    point_clouds = []
    
    for image_file in image_files:
        full_path = os.path.join(folder_path, image_file)
        print(f"Processing file: {full_path}")
        pcd = generate_point_cloud_from_depth(full_path)
        if pcd is not None:
            point_clouds.append(pcd)
            print(f"Successfully processed file: {full_path}")
        else:
            print(f"Failed to process file: {full_path}")
    
    if not point_clouds:
        raise RuntimeError("No valid point clouds were generated")
        
    # Combine all point clouds
    print("Combining point clouds")
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        combined_pcd += pcd
    
    print("Successfully combined point clouds")
    return combined_pcd

# Usage example
if __name__ == "__main__":
    try:
        print("Generating combined point cloud")
        combined_cloud = process_depth_folder("images")
        print("combined_cloud - ", combined_cloud)
        o3d.io.write_point_cloud("combined_point_cloud.ply", combined_cloud)
        print("Successfully generated combined point cloud")
    except Exception as e:
        print(f"Failed to process depth images: {str(e)}")