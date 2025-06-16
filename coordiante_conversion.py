import numpy as np

def yolo_box_to_yaw_pitch(x1, y1, x2, y2, perspective_width, perspective_height, 
                                camera_yaw, camera_pitch, camera_fov):
    """
    Convert YOLO bounding box coordinates to yaw/pitch angles in equirectangular space.
    Simple approach using direct angular mapping.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates in perspective image
        perspective_width, perspective_height: Dimensions of perspective image
        camera_yaw: Camera yaw angle in degrees (0-360)
        camera_pitch: Camera pitch angle in degrees (-90 to 90)
        camera_fov: Camera field of view in degrees
    
    Returns:
        tuple: (yaw, pitch) in degrees pointing to box center
    """
    
    # Find center of bounding box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    
    # Convert to normalized coordinates relative to image center
    # Range: [-1, 1] where 0 is center
    norm_x = (box_center_x - perspective_width/2) / (perspective_width/2)
    norm_y = (box_center_y - perspective_height/2) / (perspective_height/2)
    
    # Convert normalized coordinates to angular offsets
    # Assume square pixels and symmetric FOV
    half_fov = camera_fov / 2
    
    # Angular offset from camera center (in degrees)
    offset_yaw = norm_x * half_fov
    offset_pitch = -norm_y * half_fov  # Negative because Y increases downward in images
    
    # Add offsets to camera orientation
    target_yaw = camera_yaw + offset_yaw
    target_pitch = camera_pitch + offset_pitch
    
    # Normalize yaw to [0, 360) range
    target_yaw = target_yaw % 360
    
    # Clamp pitch to valid range
    if target_pitch > 90:
        target_pitch = 180 - target_pitch
    elif target_pitch < -90:
        target_pitch = -180 - target_pitch  
    
    return target_yaw, target_pitch


def equirectangular_to_pixel(yaw, pitch, equirect_width, equirect_height):
    """Convert yaw/pitch to equirectangular pixel coordinates."""
    u = yaw / 360.0
    v = (90 - pitch) / 180.0  # Note: 90 - pitch for typical equirectangular format
    
    x = u * equirect_width
    y = v * equirect_height
    
    return x, y


# Test both methods
if __name__ == "__main__":
    # YOLO bounding box
    x1, y1, x2, y2 = 100, 150, 300, 350
    
    # Image dimensions
    perspective_width = 640
    perspective_height = 480
    equirect_width = 2048
    equirect_height = 1024
    
    # Camera parameters
    camera_yaw = 90    # Looking right (east)
    camera_pitch = 0   # Level
    camera_fov = 60    # 60 degree FOV
    
    print("YOLO box center:", (x1+x2)/2, (y1+y2)/2)
    print("Image center:", perspective_width/2, perspective_height/2)
    print("Box offset from center:", (x1+x2)/2 - perspective_width/2, (y1+y2)/2 - perspective_height/2)
    print()
    
    # Simple method
    yaw1, pitch1 = yolo_box_to_yaw_pitch(x1, y1, x2, y2, perspective_width, perspective_height,
                                               camera_yaw, camera_pitch, camera_fov)
    print(f"Yaw={yaw1:.2f}°, Pitch={pitch1:.2f}°")
    
    # Test with center of image (should return camera yaw/pitch exactly)
    center_x, center_y = perspective_width/2, perspective_height/2
    yaw_center, pitch_center = yolo_box_to_yaw_pitch(center_x-1, center_y-1, center_x+1, center_y+1,
                                                           perspective_width, perspective_height,
                                                           camera_yaw, camera_pitch, camera_fov)
    print(f"Center test: Yaw={yaw_center:.2f}° (should be {camera_yaw}°), Pitch={pitch_center:.2f}° (should be {camera_pitch}°)")
