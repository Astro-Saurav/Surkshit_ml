"""
Main entry point for the Scream Detection System
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application
try:
    from scream_detection.gui.app import ScreamDetectionApp
    from scream_detection.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    # Try to adjust path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    # Try again
    from scream_detection.gui.app import ScreamDetectionApp
    from scream_detection.config import Config

def ensure_directories():
    """Ensure all required directories exist"""
    required_dirs = [
        os.path.join(os.path.dirname(Config.BASE_DIR), "models", "saved"),
        os.path.join(os.path.dirname(Config.BASE_DIR), "clips"),
        os.path.join(os.path.dirname(Config.BASE_DIR), "data")
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_dependencies = []
    
    
    dependencies = [
        ('numpy', 'numpy'),
        ('tensorflow', 'tensorflow'),
        ('librosa', 'librosa'),
        ('matplotlib', 'matplotlib'),
        ('sounddevice', 'sounddevice')
    ]
    
    # Check each dependency
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
        except ImportError:
            missing_dependencies.append(package_name)
    
    return missing_dependencies

def main():
    """Main application entry point"""
    # Ensure directories exist
    ensure_directories()
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("ERROR: Missing required dependencies:")
        for package in missing:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing)}")
        
        # Show GUI error if possible
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Missing Dependencies",
                f"The following required dependencies are missing:\n\n"
                f"{', '.join(missing)}\n\n"
                f"Please install them using:\n"
                f"pip install {' '.join(missing)}"
            )
            root.destroy()
        except:
            pass
        
        return 1
    
    # Start application
    try:
        root = tk.Tk()
        
        # Set application title and icon
        root.title("Scream Detection System v2.0")
        
        # Set window size and position
        window_width = 1024
        window_height = 768
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Display information about the updated system
        show_update_info = True
        # Check if this is the first run after update
        update_flag_file = os.path.join(os.path.dirname(Config.BASE_DIR), ".updated")
        if os.path.exists(update_flag_file):
            show_update_info = False
        else:
            # Create the flag file to prevent showing this message again
            with open(update_flag_file, "w") as f:
                f.write("Updated to v2.0")
        
        app = ScreamDetectionApp(root)
        
        if show_update_info:
            messagebox.showinfo(
                "System Updated",
                "The Scream Detection System has been updated with the following improvements:\n\n"
                "1. Enhanced detection sensitivity for screams\n"
                "2. Improved model architecture for better accuracy\n"
                "3. Lower default threshold (0.45) for increased sensitivity\n"
                "4. Added heuristic analysis for better classification\n\n"
                "Please check the Settings tab to adjust the threshold if needed."
            )
        
        root.mainloop()
        return 0
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())