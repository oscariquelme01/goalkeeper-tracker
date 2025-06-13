import numpy as np
import cv2

def normalize(v):
    return v / np.linalg.norm(v)

def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

def get_face_rotation(face):
    # Returns a rotation matrix from local face to global direction
    if face == "front":
        return np.eye(3)
    elif face == "right":
        return rotation_matrix_y(-np.pi / 2)  # Try negative again
    elif face == "back":
        return rotation_matrix_y(np.pi)
    elif face == "left":
        return rotation_matrix_y(np.pi / 2)   # Try positive again  
    elif face == "top":
        return rotation_matrix_x(np.pi / 2)
    elif face == "bottom":
        return rotation_matrix_x(-np.pi / 2)
    else:
        raise ValueError("Unknown face: " + face)

def pixel_to_spherical(x, y, W, H, fov_deg, face):
    # Step 1: Normalize pixel coordinates to [-1, 1]
    nx = (x / W) * 2 - 1
    ny = 1 - (y / H) * 2  # flipped Y
    
    # Step 2: Convert to 3D direction in camera space
    fov = np.radians(fov_deg)
    f = 1 / np.tan(fov / 2)
    dir_cam = np.array([nx, ny, -f])
    dir_cam = normalize(dir_cam)
    
    # Step 3: Rotate into global direction based on cubemap face
    rot = get_face_rotation(face)
    dir_global = rot @ dir_cam  # matrix multiply
    
    # Step 4: Convert to spherical coordinates
    xg, yg, zg = dir_global
    theta = np.arctan2(xg, -zg)  # azimuth
    phi = np.arcsin(yg)          # elevation
    
    return theta, phi

def perspective_from_equirect(equirect_img, theta_center, phi_center, fov_deg, out_w, out_h):
    h_equi, w_equi, *_ = equirect_img.shape  # Fixed typo here
    fov = np.radians(fov_deg)
    
    # Step 1: Create normalized image plane coordinates
    x = np.linspace(-1, 1, out_w)
    y = np.linspace(-1, 1, out_h)
    xv, yv = np.meshgrid(x, -y)  # flip Y to match image coordinates
    f = 1 / np.tan(fov / 2)
    dirs = np.stack([xv, yv, -np.ones_like(xv) * f], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)  # Normalize
    
    # Step 2: Rotate direction vectors to align with theta, phi
    def rotate(dirs, theta, phi):
        # Rotation around Y by theta (azimuth)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        # Rotation around X by phi (elevation)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi),  np.cos(phi)]
        ])
        R = Ry @ Rx  # Back to original order - let's test this first
        shape = dirs.shape
        dirs_flat = dirs.reshape(-1, 3)
        dirs_rot = dirs_flat @ R.T
        return dirs_rot.reshape(shape)
    
    dirs_world = rotate(dirs, theta_center, phi_center)
    xw, yw, zw = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
    
    # Step 3: Convert to spherical coordinates
    theta = np.arctan2(xw, -zw)
    phi = np.arcsin(yw)
    
    # Step 4: Map to equirectangular image coordinates
    u = (theta + np.pi) / (2 * np.pi) * w_equi
    v = (np.pi/2 - phi) / np.pi * h_equi
    
    # Step 5: Sample using bilinear interpolation
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    persp_img = cv2.remap(equirect_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    return persp_img
