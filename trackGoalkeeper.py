import cv2
from EquirectProcessor import EquirectProcessor
from util import pick_file_from_directory, replace_immediate_parent

# Define the 6 standard view directions (yaw, pitch in degrees)
# The first value (yaw) indexes the horizontal view (360 degrees) 0 -> front, 180 (or -180) -> back and so
# The second value (pitch) indexes the vertical view (180 degrees) 0 -> horizon, -> 90 up, -90 -> down and so 
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

# TODO: improve this!!
# Change parent dir ('equirectangular') to 'edited'
output_file = replace_immediate_parent(input_file, 'edited')

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
# TODO: to be fair OUTPUT_SIZE doesn't necesarily need to the equal to the generated perspective images
processor = EquirectProcessor(VIEWS, FOV, OUTPUT_SIZE, USE_GPU)
processor.precompute_mappings(first_frame.shape)

# Setup output video to the configured OUTPUT_SIZE
output_height, output_width = OUTPUT_SIZE
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
    
    # Process frame retrieving all the views
    views = processor.process_frame(frame)

    # TODO: write the processed view!!
    out.write(frame)
    
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
