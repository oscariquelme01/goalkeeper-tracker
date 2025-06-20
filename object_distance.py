object_sizes = {
            'person': 1.7,      # Average human height
            'sports ball': 0.22 # Average football diameter
        }

def calculate_distance(x1, y1, x2, y2, class_name):
        """
        Calculate distance using pinhole camera model
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_name: Object class name from YOLO
            
        Returns:
            Estimated distance in meters
        """
        if class_name not in object_sizes:
            return None
        
        # Get bounding box dimensions
        bbox_width = x2 - x1 # This would usually be used for objects like cars where using the width to calculate results typically yield better results 

        bbox_height = y2 - y1
        
        # Get known real-world size
        real_size = object_sizes[class_name]
        
        # Use height for most objects (more reliable than width)
        # For vehicles, use length (width in image)
        pixel_size = bbox_height
        
        # Calculate distance using pinhole camera model
        # distance = (real_size * focal_length) / pixel_size
        distance = (real_size * 320) / pixel_size
        
        return distance
