import open3d as o3d
import numpy as np
import os
from PIL import Image
import torch
import cv2
from torchvision.transforms import Compose, ToTensor, Normalize

# Load MiDaS model (ensure you have downloaded the model checkpoint)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()

# Transformation for MiDaS
transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def estimate_depth_from_rgb(image_path, target_size=(384, 384)):
    """
    Use MiDaS model to estimate depth from an RGB image
    Args:
        image_path (str): Path to the RGB image
        target_size (tuple): Target size for resizing the image
    Returns:
        numpy.ndarray: Depth map
    """
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match the input size expected by MiDaS (384x384)
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Apply the transform to the resized image
    input_tensor = transform(image_resized).unsqueeze(0)
    
    # Get depth map from MiDaS
    with torch.no_grad():
        depth_map = midas(input_tensor)
    
    # Convert depth map to numpy and normalize
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_map

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

def generate_point_cloud_from_depth(depth_path, target_size=(384, 384)):
    """
    Generate point cloud from an image
    Args:
        depth_path (str): Path to image
        target_size (tuple): Target size for resizing the image
    Returns:
        o3d.geometry.PointCloud: Generated point cloud
    """
    try:
        # Read image and estimate depth using MiDaS
        depth_np = estimate_depth_from_rgb(depth_path, target_size)
        print(f"Generated depth map for {depth_path}, shape: {depth_np.shape}")
        
        # Resize depth map to ensure consistent size
        depth_np = cv2.resize(depth_np, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Save as temporary image for Open3D
        temp_path = depth_path + '_temp.png'
        Image.fromarray(depth_np.astype(np.uint16)).save(temp_path)
        depth_image = o3d.io.read_image(temp_path)
        os.remove(temp_path)
            
        if not depth_image:
            raise RuntimeError(f"Failed to load depth image: {depth_path}")
        
        # Create camera intrinsic parameters (assuming known parameters or using defaults)
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
        
        # Remove any zero points (outliers)
        pcd = pcd.remove_non_finite_points()
        pcd = pcd[0]
        
        return pcd
        
    except Exception as e:
        print(f"Error processing {depth_path}: {str(e)}")
        return None

def process_depth_folder(folder_path, target_size=(384, 384)):
    """
    Process all images in a folder
    Args:
        folder_path (str): Path to folder containing images
        target_size (tuple): Target size for resizing the depth images
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
        pcd = generate_point_cloud_from_depth(full_path, target_size)
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
        combined_cloud = process_depth_folder("images", target_size=(384, 384))
        print(f"Combined point cloud: {combined_cloud}")
        o3d.io.write_point_cloud("combined_point_cloud.ply", combined_cloud)
        print("Successfully generated combined point cloud")
    except Exception as e:
        print(f"Failed to process depth images: {str(e)}")
