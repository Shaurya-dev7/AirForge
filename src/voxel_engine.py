"""3D Voxel grid data structure and operations."""
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Voxel:
    """Represents a single voxel."""
    color: Tuple[int, int, int]  # RGB color


class VoxelEngine:
    """Manages a 3D voxel grid."""
    
    # Color palette
    COLORS = [
        (255, 100, 50),   # Orange
        (50, 150, 255),   # Blue
        (50, 255, 100),   # Green
        (255, 50, 150),   # Pink
        (255, 255, 50),   # Yellow
        (150, 50, 255),   # Purple
        (255, 255, 255),  # White
        (100, 100, 100),  # Gray
    ]
    
    def __init__(self, grid_size: int = 16):
        """Initialize voxel engine.
        
        Args:
            grid_size: Size of the cubic grid (e.g., 16 = 16x16x16)
        """
        self.grid_size = grid_size
        self.voxels: Dict[Tuple[int, int, int], Voxel] = {}
        self.current_color_index = 0
        self.undo_stack: List[Tuple[str, Tuple[int, int, int], Optional[Voxel]]] = []
        self.max_undo = 50
        self._visible_faces_cache: Optional[List[dict]] = None
        
        print(f"[OK] Voxel engine initialized ({grid_size}^3 grid)")
    
    @property
    def current_color(self) -> Tuple[int, int, int]:
        """Get current selected color."""
        return self.COLORS[self.current_color_index]
    
    def next_color(self):
        """Cycle to next color in palette."""
        self.current_color_index = (self.current_color_index + 1) % len(self.COLORS)
    
    def prev_color(self):
        """Cycle to previous color in palette."""
        self.current_color_index = (self.current_color_index - 1) % len(self.COLORS)
    
    def is_valid_position(self, x: int, y: int, z: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z < self.grid_size
    
    def _invalidate_cache(self):
        """Invalidate the visible faces cache."""
        self._visible_faces_cache = None

    def place_voxel(self, x: int, y: int, z: int, color: Tuple[int, int, int] = None) -> bool:
        """Place a voxel at position.
        
        Args:
            x, y, z: Grid position
            color: RGB color tuple (uses current color if None)
            
        Returns:
            bool: True if placed successfully
        """
        if not self.is_valid_position(x, y, z):
            return False
        
        pos = (x, y, z)
        color = color or self.current_color
        
        # Save for undo
        old_voxel = self.voxels.get(pos)
        self._push_undo("place", pos, old_voxel)
        
        self.voxels[pos] = Voxel(color=color)
        self._invalidate_cache()
        return True
    
    def remove_voxel(self, x: int, y: int, z: int) -> bool:
        """Remove voxel at position.
        
        Args:
            x, y, z: Grid position
            
        Returns:
            bool: True if removed successfully
        """
        pos = (x, y, z)
        if pos not in self.voxels:
            return False
        
        # Save for undo
        old_voxel = self.voxels[pos]
        self._push_undo("remove", pos, old_voxel)
        
        del self.voxels[pos]
        self._invalidate_cache()
        return True
    
    def get_voxel(self, x: int, y: int, z: int) -> Optional[Voxel]:
        """Get voxel at position."""
        return self.voxels.get((x, y, z))
    
    def has_voxel(self, x: int, y: int, z: int) -> bool:
        """Check if voxel exists at position."""
        return (x, y, z) in self.voxels
    
    def _push_undo(self, action: str, pos: Tuple[int, int, int], voxel: Optional[Voxel]):
        """Push action to undo stack."""
        self.undo_stack.append((action, pos, voxel))
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
    
    def undo(self) -> bool:
        """Undo last action.
        
        Returns:
            bool: True if undo was performed
        """
        if not self.undo_stack:
            return False
        
        action, pos, voxel = self.undo_stack.pop()
        
        if action == "place":
            if voxel is None:
                # Was empty before, remove it
                if pos in self.voxels:
                    del self.voxels[pos]
            else:
                # Restore old voxel
                self.voxels[pos] = voxel
        elif action == "remove":
            # Restore removed voxel
            if voxel is not None:
                self.voxels[pos] = voxel
        
        self._invalidate_cache()
        return True
    
    def get_visible_faces(self) -> List[dict]:
        """Get list of visible voxel faces (with culling).
        
        Returns:
            List of dicts with 'pos', 'face', 'color' keys
        """
        if self._visible_faces_cache is not None:
            return self._visible_faces_cache

        faces = []
        directions = [
            ((1, 0, 0), "right"),
            ((-1, 0, 0), "left"),
            ((0, 1, 0), "top"),
            ((0, -1, 0), "bottom"),
            ((0, 0, 1), "front"),
            ((0, 0, -1), "back"),
        ]
        
        for pos, voxel in self.voxels.items():
            x, y, z = pos
            for (dx, dy, dz), face in directions:
                # Check if adjacent voxel exists (face is hidden)
                if not self.has_voxel(x + dx, y + dy, z + dz):
                    faces.append({
                        "pos": pos,
                        "face": face,
                        "color": voxel.color,
                    })
        
        self._visible_faces_cache = faces
        return faces
    
    def get_all_voxels(self) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Get all voxels as (position, color) tuples."""
        return [(pos, voxel.color) for pos, voxel in self.voxels.items()]
    
    def clear(self):
        """Remove all voxels."""
        self.voxels.clear()
        self.undo_stack.clear()
        self._invalidate_cache()
    
    def create_floor(self, y: int = 0, color: Tuple[int, int, int] = (100, 100, 100)):
        """Create a floor layer of voxels for reference.
        
        Args:
            y: Y-level for floor
            color: Floor color
        """
        for x in range(self.grid_size):
            for z in range(self.grid_size):
                self.voxels[(x, y, z)] = Voxel(color=color)
        self._invalidate_cache()
    
    def world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world position to grid coordinates.
        
        Args:
            world_pos: World space position
            
        Returns:
            Grid coordinates (clamped to valid range)
        """
        x = int(round(world_pos[0]))
        y = int(round(world_pos[1]))
        z = int(round(world_pos[2]))
        
        # Clamp to grid bounds
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        z = max(0, min(self.grid_size - 1, z))
        
        return (x, y, z)