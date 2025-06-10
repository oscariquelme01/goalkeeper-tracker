import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math

def remap_view(args):
    """Worker function for parallel remapping"""
    frame, uf, vf = args
    return cv2.remap(frame, uf, vf, cv2.INTER_LINEAR, cv2.BORDER_WRAP)

def compute_mapping_tables(equi_shape, fov_deg, pitch_deg, yaw_deg, out_res=(640, 480)):
    """Precompute mapping tables for remapping"""
    height, width = out_res
    fov = math.radians(fov_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    
    # Grid of (x, y) in normalized image space
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, -y)
    z = 1 / np.tan(fov / 2)
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm
    
    # Rotation matrices
    def rot_yaw(yaw):
        return np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0,           1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
    
    def rot_pitch(pitch):
        return np.array([
            [1, 0,            0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])
    
    rot = rot_yaw(yaw) @ rot_pitch(pitch)
    xyz = np.stack([x, y, z], axis=-1) @ rot.T
    
    # Convert to spherical coordinates
    lon = np.arctan2(xyz[..., 0], xyz[..., 2])
    lat = np.arcsin(xyz[..., 1])
    
    # Map to equirectangular coordinates
    equi_h, equi_w = equi_shape[:2]
    uf = (lon / np.pi + 1) / 2 * equi_w
    vf = (0.5 - lat / math.pi) * equi_h
    
    return uf.astype(np.float32), vf.astype(np.float32)


class EquirectProcessor:
    def __init__(self, views, fov=90, output_size=(640, 480), use_gpu=True, num_workers=None):
        self.views = views
        self.fov = fov
        self.output_size = output_size
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(cv2.cuda.getCudaEnabledDeviceCount())
        self.num_workers = num_workers or min(len(views), 6)
        self.mappings = None
        self.gpu_mappings = None
        
        if self.use_gpu:
            print(f"GPU acceleration enabled with {cv2.cuda.getCudaEnabledDeviceCount()} CUDA devices")
        else:
            print(f"Using CPU with {self.num_workers} workers")
    
    def precompute_mappings(self, equi_shape):
        """Precompute all mapping tables"""
        print("Precomputing mapping tables...")
        self.mappings = {}
        
        if self.use_gpu:
            self.gpu_mappings = {}
        
        for view_name, (yaw_deg, pitch_deg) in self.views.items():
            uf, vf = compute_mapping_tables(equi_shape, self.fov, pitch_deg, yaw_deg, self.output_size)
            self.mappings[view_name] = (uf, vf)
            
            if self.use_gpu:
                # Upload mapping tables to GPU
                gpu_uf = cv2.cuda_GpuMat()
                gpu_vf = cv2.cuda_GpuMat()
                gpu_uf.upload(uf)
                gpu_vf.upload(vf)
                self.gpu_mappings[view_name] = (gpu_uf, gpu_vf)
    
    def process_frame_gpu(self, frame):
        """Process frame using GPU acceleration"""
        views = []
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        for view_name in self.views.keys():
            gpu_uf, gpu_vf = self.gpu_mappings[view_name]
            gpu_result = cv2.cuda.remap(gpu_frame, gpu_uf, gpu_vf, cv2.INTER_LINEAR)
            
            # Download result from GPU
            result = gpu_result.download()
            views.append(result)
        
        return views
    
    def process_frame_cpu_parallel(self, frame):
        """Process frame using CPU parallelization"""
        remap_args = [(frame, uf, vf) for uf, vf in self.mappings.values()]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            views = list(executor.map(remap_view, remap_args))
        
        return views
    
    def process_frame_cpu_sequential(self, frame):
        """Process frame sequentially on CPU"""
        views = []
        for uf, vf in self.mappings.values():
            view = cv2.remap(frame, uf, vf, cv2.INTER_LINEAR, cv2.BORDER_WRAP)
            views.append(view)
        return views
    
    def process_frame(self, frame):
        """Process a single frame and return combined result"""
        if self.use_gpu:
            views = self.process_frame_gpu(frame)
        elif self.num_workers > 1:
            views = self.process_frame_cpu_parallel(frame)
        else:
            views = self.process_frame_cpu_sequential(frame)
        
        
        return views

