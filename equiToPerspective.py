import cv2
import numpy as np
from EquirectProcessor import EquirectProcessor
from util import pick_file_from_directory, replace_immediate_parent

# Define the 6 standard view directions (yaw, pitch in degrees)
VIEWS = {
    "front": (0, 0),
    "right": (90, 0),
    "back": (180, 0),
    "left": (-90, 0),
    "up": (0, 90),
    "down": (0, -90)
}

# Configuration
OUTPUT_SIZE = (640, 480)
FOV = 90
USE_GPU = True  # Set to False to use CPU parallelization
SOURCES_DIR = "sources/equirectangular"

# Get input file
input_file = pick_file_from_directory(SOURCES_DIR)
if not input_file:
    print("Failed to get input file")
    exit()

# Change parent dir ('equirectangular') to 'perspective'
output_file = replace_immediate_parent(input_file, 'perspective')

# Initialize video capture
cap = cv2.VideoCapture(input_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit()

# Initialize processor and precompute mappings
processor = EquirectProcessor(VIEWS, FOV, OUTPUT_SIZE, USE_GPU)
processor.precompute_mappings(first_frame.shape)

# Get output dimensions from first processed frame
views_sample = processor.process_frame(first_frame)

# Combine the first processed frame views
top = np.hstack(views_sample[:3])     # front, right, back
bottom = np.hstack(views_sample[3:])  # left, up, down

combined_sample = np.vstack([top, bottom])

output_height, output_width = combined_sample.shape[:2]

# Reset video and setup output
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (output_width, output_height))

print(f"Processing {total_frames} frames at {fps} FPS...")
print(f"Output dimensions: {output_width}x{output_height}")

# Process video
frame_count = 0
import time
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame retrieve all the views
    views = processor.process_frame(frame)

    # Combine views into grid
    top = np.hstack(views[:3])     # front, right, back
    bottom = np.hstack(views[3:])  # left, up, down
    combined = np.vstack([top, bottom])

    out.write(combined)
    
    frame_count += 1
    
    # Progress indicator
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        progress = (frame_count / total_frames) * 100
        eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
              f"FPS: {fps_actual:.1f} | ETA: {eta:.1f}s")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"\nCompleted in {total_time:.2f} seconds")
print(f"Average FPS: {frame_count / total_time:.2f}")
