# AirForge – Gesture‑Controlled 3D Voxel Editor

## Overview

AirForge is a lightweight, real‑time voxel editor that lets you **place, delete, rotate and colour** voxels using hand gestures captured via a webcam. The project uses:

- **MediaPipe** for hand‑landmark detection
- **OpenGL** (via PyOpenGL) for fast 3D rendering
- **Pygame** for the window and UI overlay
- A **state‑machine‑driven gesture detector** that makes interactions reliable and responsive.

## Installation

```bash
# Clone the repository (once you push it)
# git clone <repo‑url>
# cd AirForge

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` file should list `pygame`, `mediapipe`, `numpy`, `PyOpenGL` and any other packages used by the project.

## Running the Application

```bash
python -u main.py
```

The program will open a window, initialize the webcam and wait for your hand. The console prints a short log with ASCII tags (e.g. `[OK]`, `[READY]`).

## Controls & Gestures

| Gesture   | Action                               |
| --------- | ------------------------------------ |
| **Pinch** | Place a voxel at the cursor location |
| **Palm**  | Delete the voxel under the cursor    |
| **Point** | Move the cursor (hand position)      |
| **Peace** | Cycle through the colour palette     |
| **Grab**  | Rotate the camera                    |
| **Q**     | Quit the application                 |
| **Z**     | Undo last voxel operation            |
| **C**     | Change colour (alternative)          |
| **R**     | Reset view                           |

## Project Structure

```
AirForge/
├─ src/                # Core modules (hand_tracker, gesture_detector, renderer, voxel_engine, ui)
├─ tests/              # Unit‑tests for the gesture state machine
├─ main.py             # Entry point
├─ requirements.txt    # Python dependencies
├─ README.md           # This file
└─ .gitignore          # Git ignore rules
```

## Contributing

Feel free to open issues or submit pull requests. When contributing:

1. Fork the repository.
2. Create a feature branch.
3. Ensure existing tests pass (`python -m unittest discover`).
4. Follow the code‑style conventions used in the existing modules.

## License

This project is released under the MIT License.
