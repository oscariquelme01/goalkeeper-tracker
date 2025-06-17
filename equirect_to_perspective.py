import cv2
import numpy as np
from EquirectProcessor import EquirectProcessor
from util import pick_file_from_directory, replace_immediate_parent
from ultralytics import YOLO

# Define the 6 standard view directions (yaw, pitch in degrees)
VIEWS = {
    "front": (0, 0),
    "right": (90, 0),
    "back": (180, 0),
    "left": (270, 0),
    "up": (0, 90),
    "down": (0, -90)
}

# Configuration
OUTPUT_SIZE = (640, 480)
FOV = 90
USE_GPU = True  # Set to False to use CPU parallelization
SOURCES_DIR = "sources"

# Get input file
input_file = pick_file_from_directory(SOURCES_DIR)
if not input_file:
    print("Failed to get input file")
    exit()

# Change parent dir ('sources') to 'perspective'
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

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Initialize processor and precompute mappings
processor = EquirectProcessor(VIEWS, FOV, OUTPUT_SIZE, USE_GPU)
processor.precompute_mappings(first_frame.shape)

# Get output dimensions from first processed frame
views_sample = processor.process_frame(first_frame)

# Combine the first processed frame views
# front, right, back
views_sample = list(views_sample.values()) # type: ignore
top = np.hstack(views_sample[:3]) # type: ignore
# left, up, down
bottom = np.hstack(views_sample[3:]) # type: ignore

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
    views = list(views.values()) # type: ignore

    for view in views:
        results = model(view, verbose=False)
        for result in results:
            for box in result.boxes:
                if box.cls.item() == 0 and box.conf.item() > 0.75:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()

                    cv2.rectangle(view, 
                                 (int(x1), int(y1)), (int(x2), int(y2)), 
                                 (0, 255, 0), 2)
                    cv2.putText(view, f'Person {conf:.2f}',
                                (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            ) 


    # front, right, back
    top = np.hstack(views[:3]) # type: ignore
    # left, top, bottom
    bottom = np.hstack(views[3:])  # type: ignore
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
