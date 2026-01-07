"""On-screen HUD for displaying gesture and tool info."""
import pygame
from pygame.locals import *
from OpenGL.GL import *
from typing import Tuple, Optional

from .gesture_detector import Gesture


class HUD:
    """On-screen heads-up display."""
    
    # Gesture icons (emoji-style text)
    GESTURE_ICONS = {
        Gesture.NONE: "❓",
        Gesture.POINT: "👆",
        Gesture.PINCH: "🤏",
        Gesture.GRAB: "✊",
        Gesture.PALM: "🖐️",
        Gesture.PEACE: "✌️",
    }
    
    GESTURE_NAMES = {
        Gesture.NONE: "None",
        Gesture.POINT: "Point (Move)",
        Gesture.PINCH: "Pinch (Place)",
        Gesture.GRAB: "Grab (Rotate)",
        Gesture.PALM: "Palm (Delete)",
        Gesture.PEACE: "Peace (Color)",
    }
    
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize HUD.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.width = screen_width
        self.height = screen_height
        
        # Initialize pygame font
        pygame.font.init()
        try:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
        except:
            self.font = pygame.font.SysFont('arial', 28)
            self.small_font = pygame.font.SysFont('arial', 18)
    
    def resize(self, width: int, height: int):
        """Update screen dimensions."""
        self.width = width
        self.height = height
    
    def render(self, gesture: Gesture, cursor_pos: Tuple[int, int, int], 
               current_color: Tuple[int, int, int], voxel_count: int):
        """Render HUD elements.
        
        Args:
            gesture: Current detected gesture
            cursor_pos: Current 3D cursor position
            current_color: Currently selected color
            voxel_count: Number of voxels placed
        """
        # Switch to 2D rendering
        self._begin_2d()
        
        # Draw HUD background panel
        self._draw_panel(10, 10, 250, 140, (20, 20, 30, 180))
        
        # Gesture status
        gesture_text = f"Gesture: {self.GESTURE_NAMES.get(gesture, 'Unknown')}"
        self._draw_text(gesture_text, 20, 20, (255, 255, 255))
        
        # Cursor position
        pos_text = f"Cursor: ({cursor_pos[0]}, {cursor_pos[1]}, {cursor_pos[2]})"
        self._draw_text(pos_text, 20, 50, (200, 200, 200))
        
        # Current color (with swatch)
        self._draw_text("Color:", 20, 80, (200, 200, 200))
        self._draw_color_swatch(80, 80, 30, 20, current_color)
        
        # Voxel count
        count_text = f"Voxels: {voxel_count}"
        self._draw_text(count_text, 20, 110, (200, 200, 200))
        
        # Controls hint at bottom
        self._draw_panel(10, self.height - 50, 400, 40, (20, 20, 30, 150))
        controls = "Q: Quit | Z: Undo | C: Change Color | R: Reset View"
        self._draw_text(controls, 20, self.height - 40, (150, 150, 150), small=True)
        
        # Restore 3D rendering
        self._end_2d()
    
    def _begin_2d(self):
        """Switch to 2D orthographic projection."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def _end_2d(self):
        """Restore 3D projection."""
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _draw_panel(self, x: int, y: int, width: int, height: int, 
                    color: Tuple[int, int, int, int]):
        """Draw a semi-transparent panel."""
        glColor4f(color[0]/255, color[1]/255, color[2]/255, color[3]/255)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
    
    def _draw_color_swatch(self, x: int, y: int, width: int, height: int,
                           color: Tuple[int, int, int]):
        """Draw a color swatch."""
        # Background (border)
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(x-1, y-1)
        glVertex2f(x + width+1, y-1)
        glVertex2f(x + width+1, y + height+1)
        glVertex2f(x-1, y + height+1)
        glEnd()
        
        # Color fill
        glColor3f(color[0]/255, color[1]/255, color[2]/255)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
    
    def _draw_text(self, text: str, x: int, y: int, color: Tuple[int, int, int], 
                   small: bool = False):
        """Draw text using pygame surface converted to OpenGL texture."""
        font = self.small_font if small else self.font
        
        # Render text to pygame surface
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        # Draw as OpenGL texture
        width, height = text_surface.get_size()
        
        # Create texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Enable texturing
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        
        # Draw textured quad (flipped because of coordinate system)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + width, y)
        glTexCoord2f(1, 0); glVertex2f(x + width, y + height)
        glTexCoord2f(0, 0); glVertex2f(x, y + height)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([texture_id])
