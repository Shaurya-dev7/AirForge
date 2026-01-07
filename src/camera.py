"""3D camera for orbiting around the voxel scene."""
import numpy as np
from typing import Tuple


class Camera:
    """Orbiting 3D camera with perspective projection."""
    
    def __init__(self, target: Tuple[float, float, float] = (8, 8, 8)):
        """Initialize camera.
        
        Args:
            target: Point to orbit around (center of voxel grid)
        """
        self.target = np.array(target, dtype=np.float32)
        self.distance = 35.0  # Distance from target
        self.yaw = 45.0       # Horizontal rotation (degrees)
        self.pitch = 30.0     # Vertical rotation (degrees)
        
        # Smooth camera movement
        self.target_yaw = self.yaw
        self.target_pitch = self.pitch
        self.smoothing = 0.15
        
        # Limits
        self.min_pitch = -80.0
        self.max_pitch = 80.0
        self.min_distance = 10.0
        self.max_distance = 100.0
    
    def orbit(self, delta_yaw: float, delta_pitch: float):
        """Rotate camera around target.
        
        Args:
            delta_yaw: Horizontal rotation delta (degrees)
            delta_pitch: Vertical rotation delta (degrees)
        """
        self.target_yaw += delta_yaw
        self.target_pitch += delta_pitch
        
        # Clamp pitch
        self.target_pitch = max(self.min_pitch, min(self.max_pitch, self.target_pitch))
    
    def zoom(self, delta: float):
        """Zoom in/out.
        
        Args:
            delta: Positive = zoom in, negative = zoom out
        """
        self.distance -= delta
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
    
    def update(self):
        """Smooth camera movement - call each frame."""
        self.yaw += (self.target_yaw - self.yaw) * self.smoothing
        self.pitch += (self.target_pitch - self.pitch) * self.smoothing
    
    def get_position(self) -> np.ndarray:
        """Get camera position in world space.
        
        Returns:
            np.ndarray: (x, y, z) camera position
        """
        # Convert spherical to cartesian coordinates
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        x = self.distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        y = self.distance * np.sin(pitch_rad)
        z = self.distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        
        return self.target + np.array([x, y, z])
    
    def get_view_matrix(self) -> np.ndarray:
        """Get 4x4 view matrix for OpenGL.
        
        Returns:
            np.ndarray: 4x4 view matrix
        """
        eye = self.get_position()
        center = self.target
        up = np.array([0.0, 1.0, 0.0])
        
        return self._look_at(eye, center, up)
    
    def _look_at(self, eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create look-at view matrix.
        
        Args:
            eye: Camera position
            center: Look-at target
            up: Up vector
            
        Returns:
            np.ndarray: 4x4 view matrix
        """
        f = center - eye
        f = f / np.linalg.norm(f)  # Forward
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)  # Side
        
        u = np.cross(s, f)  # Up
        
        result = np.identity(4, dtype=np.float32)
        result[0, 0:3] = s
        result[1, 0:3] = u
        result[2, 0:3] = -f
        result[0, 3] = -np.dot(s, eye)
        result[1, 3] = -np.dot(u, eye)
        result[2, 3] = np.dot(f, eye)
        
        return result
    
    @staticmethod
    def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix.
        
        Args:
            fov: Field of view in degrees
            aspect: Aspect ratio (width/height)
            near: Near clipping plane
            far: Far clipping plane
            
        Returns:
            np.ndarray: 4x4 projection matrix
        """
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = f / aspect
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = (2 * far * near) / (near - far)
        result[3, 2] = -1.0
        
        return result
