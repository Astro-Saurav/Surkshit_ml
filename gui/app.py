"""
Main application class for the Scream Detection System GUI
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

from ..config import Config
from .file_analysis import FileAnalysisTab
from .realtime_monitor import RealtimeMonitorTab
from .training import TrainingTab
from .settings import SettingsTab

class ScreamDetectionApp:
    def __init__(self, root):
        """
        Initialize the Scream Detection System application
        
        Args:
            root (tk.Tk): Root tkinter window
        """
        self.root = root
        self.root.title("Scream Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Apply theme
        self.setup_theme()
        
        # Load config
        Config.load()
        
        # Create main components
        self.create_widgets()
        
        # Check for model
        self.check_model()
    
    def setup_theme(self):
        """Set up application theme and styling"""
        # Try to use a more modern theme if available
        try:
            assets_path = os.path.join(os.path.dirname(Config.BASE_DIR), "assets", "theme", "azure.tcl")
            if os.path.exists(assets_path):
                self.root.tk.call("source", assets_path)
                self.root.tk.call("set_theme", "light")
        except tk.TclError as e:
            print(f"Theme error: {e}")
            # Fall back to default styling
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
        
        # Configure custom styles
        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("TLabel", padding=2)
        style.configure("TFrame", background="#f0f0f0")
        
        # Add Accent.TButton style
        style.configure("Accent.TButton", 
                       background="#4285f4", 
                       foreground="white", 
                       padding=6)
    
    def create_widgets(self):
        """Create main application widgets"""
        # Main frame - removed padding to resolve geometry issues
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Scream Detection System", font=("Arial", 20, "bold"))
        title_label.pack(pady=10)
        
        # Tabs - create the notebook first
        self.tab_control = ttk.Notebook(main_frame)
        
        # Create tabs
        self.file_analysis_tab = FileAnalysisTab(self.tab_control)
        self.realtime_tab = RealtimeMonitorTab(self.tab_control)
        self.training_tab = TrainingTab(self.tab_control)
        self.settings_tab = SettingsTab(self.tab_control)
        
        # Add tabs to notebook
        self.tab_control.add(self.file_analysis_tab, text="File Analysis")
        self.tab_control.add(self.realtime_tab, text="Real-time Monitoring")
        self.tab_control.add(self.training_tab, text="Train Model")
        self.tab_control.add(self.settings_tab, text="Settings")
        
        # Pack notebook after adding tabs
        self.tab_control.pack(expand=1, fill="both")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Register settings update callback
        self.settings_tab.register_callback(self.on_settings_updated)
        
        # Bind tab change event
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def check_model(self):
        """Check if a trained model exists and notify user if not"""
        if not os.path.exists(Config.MODEL_PATH):
            self.status_var.set("No trained model found. Please train a model first.")
            
            # Show message after a short delay to let the GUI initialize
            self.root.after(500, lambda: messagebox.showinfo(
                "Model Not Found",
                "No trained model found. Please go to the 'Train Model' tab to train a model, "
                "or import an existing model in the 'Settings' tab."
            ))
            
            # Switch to training tab
            self.root.after(600, lambda: self.tab_control.select(2))  # Select Training tab
        else:
            self.status_var.set("Model loaded successfully")
            
            # Share the model path with tabs
            self.file_analysis_tab.set_model_path(Config.MODEL_PATH)
            self.realtime_tab.set_model_path(Config.MODEL_PATH)
    
    def on_settings_updated(self):
        """Handle settings updates"""
        # Update model path in other tabs
        self.file_analysis_tab.set_model_path(Config.MODEL_PATH)
        self.realtime_tab.set_model_path(Config.MODEL_PATH)
        
        # Update threshold
        self.file_analysis_tab.set_threshold(Config.THRESHOLD)
        self.realtime_tab.set_threshold(Config.THRESHOLD)
        
        # Update status
        self.status_var.set("Settings updated")
    
    def on_tab_changed(self, event):
        """Handle tab changed event"""
        selected_tab = self.tab_control.select()
        tab_text = self.tab_control.tab(selected_tab, "text")
        
        # Update status bar
        self.status_var.set(f"Switched to {tab_text}")
        
        # If switching to real-time tab, refresh devices
        if tab_text == "Real-time Monitoring":
            self.realtime_tab.refresh_devices()