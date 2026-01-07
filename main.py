"""
AirForge: Gesture-Controlled 3D Voxel Editor
Build voxel structures using hand gestures tracked via webcam.

Controls:
- Point (☝️): Move 3D cursor
- Pinch (🤏): Place voxel
- Open Palm (🖐️): Delete voxel
- Grab/Fist (✊): Rotate camera
- Peace (✌️): Cycle color

Keyboard:
- Q/ESC: Quit
- Z: Undo
- C: Next color
- R: Reset camera
- X: Clear all voxels
"""

import pygame
from pygame.locals import *
import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.hand_tracker import HandTracker
from src.gesture_detector import GestureDetector, Gesture
from src.voxel_engine import VoxelEngine
from src.camera import Camera
from src.renderer import Renderer
from src.ui import HUD


class AirForge:
    """Main application class."""
    
    def __init__(self, grid_size: int = 16, window_size: tuple = (1280, 720)):
        """Initialize AirForge.
        
        Args:
            grid_size: Voxel grid size
            window_size: Window dimensions (width, height)
        """
        print("\n[START] Starting AirForge - Gesture-Controlled 3D Voxel Editor\n")
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.gesture_detector = GestureDetector()
        self.voxel_engine = VoxelEngine(grid_size=grid_size)
        self.camera = Camera(target=(grid_size/2, grid_size/2, grid_size/2))
        self.renderer = Renderer(width=window_size[0], height=window_size[1])
        self.hud = HUD(window_size[0], window_size[1])
        
        self.grid_size = grid_size
        self.running = True
        
        # 3D cursor state
        self.cursor_pos = (grid_size // 2, grid_size // 2, grid_size // 2)
        self.last_hand_pos = None
        
        # Camera rotation state
        self.rotating = False
        self.last_grab_pos = None
        
        # Action cooldowns to prevent spam
        self.place_cooldown = 0
        self.delete_cooldown = 0
        self.cooldown_frames = 10
        
        # Add some starter voxels for demo
        self._create_demo_structure()
        
        print("\n[READY] AirForge ready! Show your hand to the camera.\n")
    
    def _create_demo_structure(self):
        """Create a small demo structure."""
        center = self.grid_size // 2
        # Small platform
        for x in range(center-2, center+3):
            for z in range(center-2, center+3):
                self.voxel_engine.place_voxel(x, 0, z, (100, 100, 100))
    
    def run(self):
        """Main application loop."""
        clock = pygame.time.Clock()
        
        try:
            while self.running:
                # Handle events
                self._handle_events()
                
                # Process hand tracking
                frame, landmarks = self.hand_tracker.process()
                if frame is None:
                    continue
                
                # Detect gesture
                gesture = self.gesture_detector.detect(landmarks)
                
                # Update cursor and actions based on gesture
                self._process_gesture(gesture, landmarks)
                
                # Update camera
                self.camera.update()
                
                # Decay cooldowns
                if self.place_cooldown > 0:
                    self.place_cooldown -= 1
                if self.delete_cooldown > 0:
                    self.delete_cooldown -= 1
                
                # Render
                # Draw landmarks on frame for "trace" effect
                if landmarks:
                    self.hand_tracker.draw_landmarks(frame, landmarks)
                
                # Convert BGR to RGB for rendering
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self._render(gesture, frame_rgb)
                
                # Cap framerate
                clock.tick(60)
        
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user")
        
        finally:
            self._cleanup()
    
    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            
            elif event.type == VIDEORESIZE:
                self.renderer.handle_resize(event.w, event.h)
                self.hud.resize(event.w, event.h)
            
            elif event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    self.running = False
                elif event.key == K_z:
                    self.voxel_engine.undo()
                elif event.key == K_c:
                    self.voxel_engine.next_color()
                elif event.key == K_r:
                    # Reset camera
                    self.camera.yaw = 45
                    self.camera.pitch = 30
                    self.camera.target_yaw = 45
                    self.camera.target_pitch = 30
                elif event.key == K_x:
                    self.voxel_engine.clear()
                    self._create_demo_structure()
    
    def _process_gesture(self, gesture: Gesture, landmarks):
        """Process detected gesture and update state.
        
        Args:
            gesture: Detected gesture
            landmarks: Hand landmarks
        """
        if landmarks is None:
            self.last_hand_pos = None
            self.last_grab_pos = None
            return
        
        # Get hand position
        hand_pos = self.gesture_detector.get_index_tip_position(landmarks)
        if hand_pos is None:
            return
        
        # Map hand position to 3D cursor
        # Hand x (0-1) -> Grid x (0 to grid_size)
        # Hand y (0-1) -> Grid y (grid_size to 0) - inverted
        # Hand z (depth) -> Grid z (approximate)
        cursor_x = int(hand_pos[0] * self.grid_size)
        cursor_y = int((1 - hand_pos[1]) * self.grid_size)  # Invert Y
        
        # Use hand depth for Z, scaled and offset
        # MediaPipe z is negative when hand is closer
        cursor_z = int(self.grid_size / 2 - hand_pos[2] * 50)
        
        # Clamp to grid
        cursor_x = max(0, min(self.grid_size - 1, cursor_x))
        cursor_y = max(0, min(self.grid_size - 1, cursor_y))
        cursor_z = max(0, min(self.grid_size - 1, cursor_z))
        
        self.cursor_pos = (cursor_x, cursor_y, cursor_z)
        
        # Context Locking (Time-based)
        current_time = pygame.time.get_ticks()
        if not hasattr(self, 'action_lock_until'):
            self.action_lock_until = 0
            self.pinch_fired = False
            self.delete_fired = False
        
        # Reset firing flags if gesture changes
        if gesture != Gesture.PINCH:
            self.pinch_fired = False
        if gesture != Gesture.PALM:
            self.delete_fired = False
            
        # Block actions if locked
        if current_time < self.action_lock_until:
            return
        
        # Handle gesture actions
        if gesture == Gesture.PINCH:
            # Place voxel (One-shot)
            if not self.pinch_fired:
                if self.voxel_engine.place_voxel(*self.cursor_pos):
                    # Success
                    self.pinch_fired = True
                    # Lock for 200ms to prevent double triggers
                    self.action_lock_until = current_time + 200 
        
        elif gesture == Gesture.PALM:
            # Delete voxel (Continuous but throttled or One-shot?)
            # User guideline: "No PLACE during RELEASE". Handled by state machine.
            # Let's make Delete One-Shot per "palm open" to be safe, or slow repeat.
            # User didn't specify strict one-shot for delete, but "Context Locking".
            # Let's use 150ms repeat rate
            if not self.delete_fired or (current_time > self.action_lock_until):
                 if self.voxel_engine.remove_voxel(*self.cursor_pos):
                     self.delete_fired = True
                     self.action_lock_until = current_time + 150
        
        elif gesture == Gesture.GRAB:
            # Rotate camera
            if self.last_grab_pos is not None:
                dx = (hand_pos[0] - self.last_grab_pos[0]) * 200
                dy = (hand_pos[1] - self.last_grab_pos[1]) * 200
                self.camera.orbit(-dx, dy)
            self.last_grab_pos = hand_pos
        else:
            self.last_grab_pos = None
        
        if gesture == Gesture.PEACE:
            # Color picker - cycle on gesture start (already one-shot via flag)
            if not hasattr(self, '_peace_active') or not self._peace_active:
                self.voxel_engine.next_color()
                self._peace_active = True
                self.action_lock_until = current_time + 300 # Lock after color switch
        else:
            self._peace_active = False
        
        self.last_hand_pos = hand_pos
    
    def _render(self, gesture: Gesture, bg_frame=None):
        """Render the scene.
        
        Args:
            gesture: Current gesture for HUD display
            bg_frame: Optional RGB background frame
        """
        # Clear/Draw Background
        if bg_frame is not None:
            self.renderer.render_background(bg_frame)
        else:
            self.renderer.clear()
        
        # Set camera
        self.renderer.set_camera(self.camera)
        
        # Render grid floor
        self.renderer.render_grid_floor(self.grid_size)
        
        # Render axes
        self.renderer.render_axes()
        
        # Render voxels
        self.renderer.render_voxels(self.voxel_engine)
        
        # Render 3D cursor
        cursor_color = self.voxel_engine.current_color
        self.renderer.render_cursor(self.cursor_pos, cursor_color)
        
        # Render HUD
        self.hud.render(
            gesture=gesture,
            cursor_pos=self.cursor_pos,
            current_color=self.voxel_engine.current_color,
            voxel_count=len(self.voxel_engine.voxels)
        )
        
        # Swap buffers
        self.renderer.swap()
    
    def _cleanup(self):
        """Clean up resources."""
        print("\n[STOP] Shutting down AirForge...")
        self.hand_tracker.release()
        self.renderer.quit()
        cv2.destroyAllWindows()
        print("[DONE] Goodbye!\n")


def main():
    """Entry point."""
    app = AirForge(grid_size=16, window_size=(1280, 720))
    app.run()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())