import cv2
import os

from util import pick_file_from_directory

def extract_frame(video_path, frame_number):
    """
    Extract a specific frame from a video file.
    
    Args:
        video_path (str): Path to the input video file
        frame_number (int): Frame number to extract (0-based)
        output_path (str, optional): Path to save the extracted frame. 
                                   If None, will save as 'frame_{frame_number}.jpg'
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Check if frame number is valid
    if frame_number >= total_frames or frame_number < 0:
        print(f"Error: Frame number {frame_number} is out of range (0-{total_frames-1})")
        cap.release()
        return False
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return False
    
    # Save the frame
    success = cv2.imshow(f'Frame number {frame_number}', frame)
    
    if success:
        print(f"Frame {frame_number} extracted successfully!")
        print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
    
    # Cleanup
    cap.release()
    return success

def extract_multiple_frames(video_path, frame_numbers, output_dir="extracted_frames"):
    """
    Extract multiple frames from a video file.
    
    Args:
        video_path (str): Path to the input video file
        frame_numbers (list): List of frame numbers to extract
        output_dir (str): Directory to save extracted frames
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_num in frame_numbers:
        extract_frame(video_path, frame_num)

    cv2.waitKey(0)  # Wait for any key press


# Example usage
if __name__ == "__main__":
    # Example 1: Extract a single frame
    video_file = pick_file_from_directory('perspective')

    frame_to_extract = int(input('Pick frame to extract: '))
    extract_frame(video_file, frame_to_extract)
    cv2.waitKey(0)

    # extract_multiple_frames(video_file, [131, 132, 133, 134])
