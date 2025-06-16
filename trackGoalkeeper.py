import cv2
from EquirectProcessor import EquirectProcessor, remap_single_view
from coordiante_conversion import yolo_box_to_yaw_pitch
from object_distance import calculate_distance
from util import pick_file_from_directory, replace_immediate_parent
from ultralytics import YOLO

# Define the 6 standard view directions (yaw, pitch in degrees)
# The first value (yaw) indexes the horizontal view (360 degrees) 0 -> front, 180 (or -180) -> back and so
# The second value (pitch) indexes the vertical view (180 degrees) 0 -> horizon, -> 90 up, -90 -> down and so 
VIEWS = {
    "right": (90, 0),
    "back": (180, 0),
    "left": (-90, 0),
    "front": (0, 0),
    "bottom": (0, -90),
    "top": (0, 90)
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

# Initialize YOLO model
# TODO: send this to GPU
model = YOLO('yolov8n.pt')

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

# Default output for the first frames before the person is detected
default_view_name = 'front'
last_output_frame = None
output_frame = None

# Save the yaw and pitch calculated from the previous frame to ensure that there are no re-writes of the same frame
last_yaw, last_pitch = 0,0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    person_detected = False
    
    # Process frame retrieving all the views
    views = processor.process_frame(frame)
    
    for view_name, view in views.items(): #type: ignore
        if person_detected:  # Only process first detection
            break
            
        results = model(view, verbose=False)
        for result in results:
            for box in result.boxes:
                if box.cls.item() == 0 and box.conf.item() > 0.7:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    yaw, pitch = yolo_box_to_yaw_pitch(x1, y1, x2, y2, output_width, output_height, VIEWS[view_name][0], VIEWS[view_name][1], FOV)
                    last_yaw, last_pitch = yaw, pitch
                    
                    # Generate the detection-centered view
                    output_frame = remap_single_view(frame, yaw, -pitch, FOV, OUTPUT_SIZE)

                    # Re-run detection on the new centered view to get correct coordinates
                    new_results = model(output_frame, verbose=False)
                    for new_result in new_results:
                        for new_box in new_result.boxes:
                            if new_box.cls.item() == 0 and new_box.conf.item() > 0.7:  # Person detected in new view
                                nx1, ny1, nx2, ny2 = new_box.xyxy[0].cpu().numpy()
                                new_conf = new_box.conf.item()

                                distance = calculate_distance(nx1, ny1, nx2, ny2, model.names[new_box.cls.item()])
                                print(f'Found person in view {view_name} with confidence {new_box.conf.item()}.2f at frame {frame_count} and distance {distance}. Yaw: {yaw}deg Pitch: {pitch}deg')
                                
                                # Draw bounding box on the new output frame
                                cv2.rectangle(output_frame, 
                                             (int(nx1), int(ny1)), (int(nx2), int(ny2)), 
                                             (0, 255, 0), 2)
                                cv2.putText(output_frame, f'Person {new_conf:.2f} @ {distance}', 
                                           (int(nx1), int(ny1-10)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                break

                    person_detected = True
                    break
            
            if person_detected:
                break
    
    # If no person detected, use the default view
    if not person_detected:
        output_frame = remap_single_view(frame, last_yaw, -last_pitch) if last_output_frame is not None else views[default_view_name] #type: ignore
        
        no_person_detected_debug_str = f'last output frame with yaw {last_yaw}deg & pitch {last_pitch}deg' if last_output_frame is not None else f'default output frame: {default_view_name}'
        print('no person detected, defaulting to ' + no_person_detected_debug_str)
    
    # Always write exactly one perspective frame
    out.write(output_frame)
    last_output_frame = output_frame
    
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
