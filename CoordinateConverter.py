import numpy as np

def perspective_to_equirectangular(x, y, 
                                 perspective_width, perspective_height,
                                 equirect_width, equirect_height,
                                 center_theta=0, center_phi=np.pi/2, 
                                 fov_horizontal=np.pi/3):
    """
    Convert perspective view coordinates to equirectangular coordinates.
    
    Args:
        x, y: Pixel coordinates in perspective view
        perspective_width, perspective_height: Size of perspective view
        equirect_width, equirect_height: Size of equirectangular image
        center_theta: Horizontal center direction in radians (0 = front, π = back)
        center_phi: Vertical center direction in radians (π/2 = horizon, 0 = up, π = down)
        fov_horizontal: Horizontal field of view in radians
    
    Returns:
        (u, v): Equirectangular pixel coordinates
    """
    
    # Step 1: Convert perspective pixels to normalized coordinates (-1 to 1)
    x_norm = (x - perspective_width/2) / (perspective_width/2)
    y_norm = (y - perspective_height/2) / (perspective_height/2)
    
    # Step 2: Calculate field of view
    fov_vertical = fov_horizontal * (perspective_height / perspective_width)
    
    # Step 3: Convert to viewing angles relative to center
    theta_offset = x_norm * (fov_horizontal / 2)
    phi_offset = y_norm * (fov_vertical / 2)
    
    # Step 4: Add to center direction to get absolute spherical coordinates
    theta = center_theta + theta_offset
    phi = center_phi - phi_offset  # Subtract because y increases downward
    
    # Step 5: Normalize theta to [0, 2π] range
    theta = theta % (2 * np.pi)
    
    # Step 6: Clamp phi to [0, π] range
    phi = np.clip(phi, 0, np.pi)
    
    # Step 7: Convert spherical coordinates to equirectangular pixels
    u = (theta / (2 * np.pi)) * equirect_width
    v = (phi / np.pi) * equirect_height
    
    return u, v
