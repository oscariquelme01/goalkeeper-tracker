import os
from pathlib import Path

def pick_file_from_directory(directory_path="."):
    """
    Reads a directory and allows user to pick a file through CLI.
    
    Args:
        directory_path (str): Path to the directory to read. Defaults to current directory.
    
    Returns:
        str: Full path to the selected file, or None if cancelled/no files found.
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist.")
            return None
        
        if not os.path.isdir(directory_path):
            print(f"Error: '{directory_path}' is not a directory.")
            return None
        
        # Get all files in the directory (not subdirectories)
        files = [f for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f))]
        
        if not files:
            print(f"No files found in directory '{directory_path}'.")
            return None
        
        # Sort files alphabetically
        files.sort()
        
        # Display files with numbers
        print(f"\nFiles in '{directory_path}':")
        print("-" * 40)
        for i, file in enumerate(files, 1):
            print(f"{i:2d}. {file}")
        print("-" * 40)
        print(f"{len(files) + 1:2d}. Cancel")
        
        # Get user selection
        while True:
            try:
                choice = input(f"\nSelect a file (1-{len(files)}) or {len(files) + 1} to cancel: ")
                choice_num = int(choice)
                
                if choice_num == len(files) + 1:
                    print("Selection cancelled.")
                    return None
                elif 1 <= choice_num <= len(files):
                    selected_file = files[choice_num - 1]
                    full_path = os.path.join(directory_path, selected_file)
                    print(f"Selected: {selected_file}")
                    return full_path
                else:
                    print(f"Please enter a number between 1 and {len(files) + 1}.")
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def replace_immediate_parent(path_str: str, new_dir: str) -> str:
    """
    Replace the immediate parent directory of a file with `new_dir`.
    
    Args:
        path_str (str): The original file path.
        new_dir (str): The new directory name.
    
    Returns:
        str: The updated path string.
    """
    original_path = Path(path_str)
    
    # Build the new path
    new_parent = original_path.parent.parent / new_dir
    new_path = new_parent / original_path.name
    return str(new_path)
