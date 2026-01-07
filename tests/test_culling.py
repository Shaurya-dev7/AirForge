
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.voxel_engine import VoxelEngine

def test_culling():
    engine = VoxelEngine(grid_size=16)
    
    print("Test 1: Single Voxel")
    engine.place_voxel(5, 5, 5)
    faces = engine.get_visible_faces()
    print(f"Faces: {len(faces)} (Expected: 6)")
    assert len(faces) == 6
    
    print("\nTest 2: Two adjacent voxels")
    engine.place_voxel(6, 5, 5)
    faces = engine.get_visible_faces()
    # 2 voxels, 1 shared face hidden on both sides = 12 - 2 = 10
    print(f"Faces: {len(faces)} (Expected: 10)")
    assert len(faces) == 10
    
    print("\nTest 3: 3x3x3 Cube")
    # Fill 3x3x3 area
    for x in range(3):
        for y in range(3):
            for z in range(3):
                engine.place_voxel(x, y, z)
    
    faces = engine.get_visible_faces()
    # Outer surface area of 3x3x3 cube is 6 faces * (3*3) = 54
    print(f"Faces: {len(faces)} (Expected: 54)")
    assert len(faces) == 54

    print("\n✅ All culling tests passed!")

if __name__ == "__main__":
    test_culling()
