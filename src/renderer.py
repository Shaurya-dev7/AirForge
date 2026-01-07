"""OpenGL renderer for voxels and UI."""
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import Tuple, List, Optional

from .camera import Camera
from .voxel_engine import VoxelEngine


class Renderer:
    """OpenGL renderer for the voxel editor."""
    
    def __init__(self, width: int = 1280, height: int = 720, title: str = "AirForge | Voxel Editor"):
        """Initialize renderer.
        
        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        pygame.init()
        pygame.display.set_caption(title)
        
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
        
        # OpenGL setup
        self._setup_opengl()
        
        # Font for HUD (we'll use OpenGL text later)
        pygame.font.init()
        
        print(f"[OK] Renderer initialized ({width}x{height})")
    
    def _setup_opengl(self):
        """Configure OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, [20.0, 30.0, 20.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Background color (dark gray)
        glClearColor(0.1, 0.1, 0.15, 1.0)
        
        # Update viewport
        self._update_projection()
        
        # Background texture state
        self.bg_texture = None

    def render_background(self, image_data: np.ndarray):
        """Render image as background.
        
        Args:
            image_data: RGB numpy array
        """
        # Create texture if not exists
        if self.bg_texture is None:
            self.bg_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.bg_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        h, w, _ = image_data.shape
        
        # Select our texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        
        # Upload data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        
        # Save state
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_TRANSFORM_BIT)
        
        # Switch to 2D projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting/depth for background
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glColor3f(1, 1, 1)  # White modulation to show original colors
        
        # Draw full screen quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)  # Top-Left (image is top-down)
        glTexCoord2f(1, 1); glVertex2f(1, 0)  # Top-Right
        glTexCoord2f(1, 0); glVertex2f(1, 1)  # Bottom-Right
        glTexCoord2f(0, 0); glVertex2f(0, 1)  # Bottom-Left
        glEnd()
        
        # Restore state
        glPopMatrix() # Modelview
        glMatrixMode(GL_PROJECTION)
        glPopMatrix() # Projection
        glPopAttrib()
        
        glDisable(GL_TEXTURE_2D)
        
        # Clear depth buffer so 3D objects draw over it correctly
        glClear(GL_DEPTH_BUFFER_BIT)
    
    def _update_projection(self):
        """Update projection matrix for current window size."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
    
    def handle_resize(self, width: int, height: int):
        """Handle window resize.
        
        Args:
            width: New width
            height: New height
        """
        self.width = width
        self.height = max(height, 1)  # Prevent division by zero
        glViewport(0, 0, self.width, self.height)
        self._update_projection()
    
    def clear(self):
        """Clear the screen."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    def set_camera(self, camera: Camera):
        """Apply camera transformation.
        
        Args:
            camera: Camera object
        """
        glLoadIdentity()
        
        pos = camera.get_position()
        target = camera.target
        
        gluLookAt(
            pos[0], pos[1], pos[2],
            target[0], target[1], target[2],
            0, 1, 0
        )
    
    def render_voxels(self, voxel_engine: VoxelEngine):
        """Render all voxels using face culling.
        
        Args:
            voxel_engine: VoxelEngine with voxels to render
        """
        # Use cached visible faces for performance
        faces = voxel_engine.get_visible_faces()
        
        glBegin(GL_QUADS)
        for face_data in faces:
            self._draw_face_primitive(face_data['pos'], face_data['face'], face_data['color'])
        glEnd()
        
        # Draw wireframes in a second pass (optional, but looks nice)
        # For performance on large grids, you might want to disable this or only draw selected
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for face_data in faces:
             self._draw_face_outline_primitive(face_data['pos'], face_data['face'], face_data['color'])
        glEnd()

    def _draw_face_primitive(self, pos: Tuple[int, int, int], face: str, color: Tuple[int, int, int]):
        """Draw a single face quad (must be called inside glBegin(GL_QUADS))."""
        x, y, z = pos
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        glColor3f(r, g, b)
        
        # Vertices relative to center (0,0,0) -> (x+0.5, y+0.5, z+0.5) translation
        # To avoid constant glPushMatrix/pop, we just add the offset directly
        # Grid cells are size 1.0, centered at x+0.5
        
        # Coordinates
        x0, x1 = x, x + 1
        y0, y1 = y, y + 1
        z0, z1 = z, z + 1
        
        if face == "front":
            glNormal3f(0, 0, 1)
            glVertex3f(x0, y0, z1)
            glVertex3f(x1, y0, z1)
            glVertex3f(x1, y1, z1)
            glVertex3f(x0, y1, z1)
        
        elif face == "back":
            glNormal3f(0, 0, -1)
            glVertex3f(x0, y0, z0)
            glVertex3f(x0, y1, z0)
            glVertex3f(x1, y1, z0)
            glVertex3f(x1, y0, z0)
            
        elif face == "top":
            glNormal3f(0, 1, 0)
            glVertex3f(x0, y1, z0)
            glVertex3f(x0, y1, z1)
            glVertex3f(x1, y1, z1)
            glVertex3f(x1, y1, z0)
            
        elif face == "bottom":
            glNormal3f(0, -1, 0)
            glVertex3f(x0, y0, z0)
            glVertex3f(x1, y0, z0)
            glVertex3f(x1, y0, z1)
            glVertex3f(x0, y0, z1)
            
        elif face == "right":
            glNormal3f(1, 0, 0)
            glVertex3f(x1, y0, z0)
            glVertex3f(x1, y1, z0)
            glVertex3f(x1, y1, z1)
            glVertex3f(x1, y0, z1)
            
        elif face == "left":
            glNormal3f(-1, 0, 0)
            glVertex3f(x0, y0, z0)
            glVertex3f(x0, y0, z1)
            glVertex3f(x0, y1, z1)
            glVertex3f(x0, y1, z0)

    def _draw_face_outline_primitive(self, pos: Tuple[int, int, int], face: str, color: Tuple[int, int, int]):
        """Draw outline for a single face (must be called inside glBegin(GL_LINES))."""
        x, y, z = pos
        # Darker color for outline
        r, g, b = color[0] / 255.0 * 0.5, color[1] / 255.0 * 0.5, color[2] / 255.0 * 0.5
        glColor3f(r, g, b)
        
        # Slight epsilon to prevent z-fighting
        e = 0.001
        x0, x1 = x - e, x + 1 + e
        y0, y1 = y - e, y + 1 + e
        z0, z1 = z - e, z + 1 + e
        
        if face == "front":
            glVertex3f(x0, y0, z1); glVertex3f(x1, y0, z1)
            glVertex3f(x1, y0, z1); glVertex3f(x1, y1, z1)
            glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
            glVertex3f(x0, y1, z1); glVertex3f(x0, y0, z1)
        elif face == "back":
            glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
            glVertex3f(x1, y0, z0); glVertex3f(x1, y1, z0)
            glVertex3f(x1, y1, z0); glVertex3f(x0, y1, z0)
            glVertex3f(x0, y1, z0); glVertex3f(x0, y0, z0)
        elif face == "top":
            glVertex3f(x0, y1, z0); glVertex3f(x0, y1, z1)
            glVertex3f(x0, y1, z1); glVertex3f(x1, y1, z1)
            glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0)
            glVertex3f(x1, y1, z0); glVertex3f(x0, y1, z0)
        elif face == "bottom":
            glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
            glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1)
            glVertex3f(x1, y0, z1); glVertex3f(x0, y0, z1)
            glVertex3f(x0, y0, z1); glVertex3f(x0, y0, z0)
        elif face == "right":
            glVertex3f(x1, y0, z0); glVertex3f(x1, y1, z0)
            glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1)
            glVertex3f(x1, y1, z1); glVertex3f(x1, y0, z1)
            glVertex3f(x1, y0, z1); glVertex3f(x1, y0, z0)
        elif face == "left":
            glVertex3f(x0, y0, z0); glVertex3f(x0, y1, z0)
            glVertex3f(x0, y1, z0); glVertex3f(x0, y1, z1)
            glVertex3f(x0, y1, z1); glVertex3f(x0, y0, z1)
            glVertex3f(x0, y0, z1); glVertex3f(x0, y0, z0)

    
    def render_cursor(self, position: Tuple[int, int, int], color: Tuple[int, int, int] = (255, 255, 0)):
        """Render 3D cursor at grid position.
        
        Args:
            position: Grid position
            color: Cursor color
        """
        x, y, z = position
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        
        glPushMatrix()
        glTranslatef(x + 0.5, y + 0.5, z + 0.5)
        
        # Draw wireframe cube for cursor
        glDisable(GL_LIGHTING)
        glColor3f(r, g, b)
        glLineWidth(2.0)
        
        s = 0.55
        glBegin(GL_LINES)
        # Bottom face
        glVertex3f(-s, -s, -s); glVertex3f(s, -s, -s)
        glVertex3f(s, -s, -s); glVertex3f(s, -s, s)
        glVertex3f(s, -s, s); glVertex3f(-s, -s, s)
        glVertex3f(-s, -s, s); glVertex3f(-s, -s, -s)
        # Top face
        glVertex3f(-s, s, -s); glVertex3f(s, s, -s)
        glVertex3f(s, s, -s); glVertex3f(s, s, s)
        glVertex3f(s, s, s); glVertex3f(-s, s, s)
        glVertex3f(-s, s, s); glVertex3f(-s, s, -s)
        # Vertical edges
        glVertex3f(-s, -s, -s); glVertex3f(-s, s, -s)
        glVertex3f(s, -s, -s); glVertex3f(s, s, -s)
        glVertex3f(s, -s, s); glVertex3f(s, s, s)
        glVertex3f(-s, -s, s); glVertex3f(-s, s, s)
        glEnd()
        
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def render_grid_floor(self, size: int = 16, y: float = -0.01):
        """Render a grid floor for reference.
        
        Args:
            size: Grid size
            y: Y position of floor
        """
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.35)
        glLineWidth(1.0)
        
        glBegin(GL_LINES)
        for i in range(size + 1):
            # X lines
            glVertex3f(0, y, i)
            glVertex3f(size, y, i)
            # Z lines
            glVertex3f(i, y, 0)
            glVertex3f(i, y, size)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def render_axes(self, size: float = 3.0):
        """Render coordinate axes at origin.
        
        Args:
            size: Length of axes
        """
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        glBegin(GL_LINES)
        # X axis - Red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(size, 0, 0)
        # Y axis - Green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, size, 0)
        # Z axis - Blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, size)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def swap(self):
        """Swap buffers to display rendered frame."""
        pygame.display.flip()
    
    def quit(self):
        """Clean up renderer."""
        pygame.quit()
        print("[OK] Renderer closed")
