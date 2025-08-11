"""
Settings tab for the Scream Detection System GUI
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from ..config import Config
from ..utils.notification import NotificationSystem

class SettingsTab(ttk.Frame):
    def __init__(self, parent):
        """
        Initialize settings tab
        
        Args:
            parent (ttk.Notebook): Parent notebook widget
        """
        super().__init__(parent, padding=10)
        
        # Initialize state variables
        self.notification = NotificationSystem()
        self.callback = None
        
        # Create UI components
        self.create_widgets()
        
        # Load current settings
        self.load_settings()
    
    def create_widgets(self):
        """Create tab widgets"""
        # Model settings
        model_frame = ttk.LabelFrame(self, text="Model Settings")
        model_frame.pack(fill=tk.X, pady=10)
        
        # Model path
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, padx=10, pady=10)
        
        model_label = ttk.Label(model_path_frame, text="Model Path:")
        model_label.pack(side=tk.LEFT, padx=5)
        
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_path_frame, textvariable=self.model_path_var, width=50)
        model_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_model_btn = ttk.Button(model_path_frame, text="Browse", command=self.browse_model)
        browse_model_btn.pack(side=tk.LEFT, padx=5)
        
        # Detection threshold
        threshold_frame = ttk.Frame(model_frame)
        threshold_frame.pack(fill=tk.X, padx=10, pady=10)
        
        threshold_label = ttk.Label(threshold_frame, text="Detection Threshold:")
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_slider = ttk.Scale(threshold_frame, from_=0.1, to=0.95, 
                                    variable=self.threshold_var, 
                                    orient=tk.HORIZONTAL)
        threshold_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.threshold_value_label = ttk.Label(threshold_frame, text="0.7")
        self.threshold_value_label.pack(side=tk.LEFT, padx=5)
        
        # Update threshold value label when slider moves
        threshold_slider.bind("<Motion>", self.update_threshold_label)
        
        # Notification settings
        notification_frame = ttk.LabelFrame(self, text="Notification Settings")
        notification_frame.pack(fill=tk.X, pady=10)
        
        # Parent phone
        phone_frame = ttk.Frame(notification_frame)
        phone_frame.pack(fill=tk.X, padx=10, pady=5)
        
        phone_label = ttk.Label(phone_frame, text="Parent Phone Number:")
        phone_label.pack(side=tk.LEFT, padx=5)
        
        self.phone_var = tk.StringVar()
        phone_entry = ttk.Entry(phone_frame, textvariable=self.phone_var, width=20)
        phone_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Parent email
        email_frame = ttk.Frame(notification_frame)
        email_frame.pack(fill=tk.X, padx=10, pady=5)
        
        email_label = ttk.Label(email_frame, text="Parent Email Address:")
        email_label.pack(side=tk.LEFT, padx=5)
        
        self.email_var = tk.StringVar()
        email_entry = ttk.Entry(email_frame, textvariable=self.email_var, width=30)
        email_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Test notification button
        test_frame = ttk.Frame(notification_frame)
        test_frame.pack(fill=tk.X, padx=10, pady=10)
        
        test_btn = ttk.Button(test_frame, text="Test Notification", command=self.test_notification)
        test_btn.pack(side=tk.LEFT, padx=5)
        
        # Audio settings
        audio_frame = ttk.LabelFrame(self, text="Audio Settings")
        audio_frame.pack(fill=tk.X, pady=10)
        
        # Sample rate
        sample_rate_frame = ttk.Frame(audio_frame)
        sample_rate_frame.pack(fill=tk.X, padx=10, pady=5)
        
        sample_rate_label = ttk.Label(sample_rate_frame, text="Sample Rate (Hz):")
        sample_rate_label.pack(side=tk.LEFT, padx=5)
        
        self.sample_rate_var = tk.IntVar(value=16000)
        sample_rate_combo = ttk.Combobox(sample_rate_frame, textvariable=self.sample_rate_var, 
                                        values=[8000, 16000, 22050, 44100, 48000], 
                                        state="readonly", width=10)
        sample_rate_combo.pack(side=tk.LEFT, padx=5)
        
        # Audio duration
        duration_frame = ttk.Frame(audio_frame)
        duration_frame.pack(fill=tk.X, padx=10, pady=5)
        
        duration_label = ttk.Label(duration_frame, text="Audio Duration (seconds):")
        duration_label.pack(side=tk.LEFT, padx=5)
        
        self.duration_var = tk.DoubleVar(value=3.0)
        duration_spinbox = ttk.Spinbox(duration_frame, from_=1.0, to=10.0, 
                                      increment=0.5, textvariable=self.duration_var, 
                                      width=10)
        duration_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Button frame
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=20)
        
        save_btn = ttk.Button(button_frame, text="Save Settings", 
                            command=self.save_settings, style="Accent.TButton")
        save_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = ttk.Button(button_frame, text="Reset to Defaults", 
                             command=self.reset_defaults)
        reset_btn.pack(side=tk.LEFT, padx=5)
    
    def register_callback(self, callback):
        """
        Register callback for settings changes
        
        Args:
            callback (function): Callback function
        """
        self.callback = callback
    
    def load_settings(self):
        """Load current settings into UI"""
        # Load model path
        self.model_path_var.set(Config.MODEL_PATH)
        
        # Load threshold
        self.threshold_var.set(Config.THRESHOLD)
        self.update_threshold_label()
        
        # Load notification settings
        self.phone_var.set(Config.PARENT_PHONE or "")
        self.email_var.set(Config.PARENT_EMAIL or "")
        
        # Load audio settings
        self.sample_rate_var.set(Config.SAMPLE_RATE)
        self.duration_var.set(Config.AUDIO_DURATION)
    
    def save_settings(self):
        """Save settings from UI to config"""
        try:
            # Update model path if changed and valid
            model_path = self.model_path_var.get()
            if model_path and model_path != Config.MODEL_PATH:
                if not os.path.exists(model_path):
                    messagebox.showwarning("Warning", 
                                          "Model file not found. Path will be saved but model won't be loaded.")
                Config.MODEL_PATH = model_path
            
            # Update detection threshold
            Config.THRESHOLD = self.threshold_var.get()
            
            # Update notification settings
            Config.PARENT_PHONE = self.phone_var.get() or None
            Config.PARENT_EMAIL = self.email_var.get() or None
            
            # Update audio settings
            Config.SAMPLE_RATE = self.sample_rate_var.get()
            Config.AUDIO_DURATION = self.duration_var.get()
            
            # Save config
            Config.save()
            
            # Call callback if registered
            if self.callback:
                self.callback()
            
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def reset_defaults(self):
        """Reset settings to defaults"""
        if messagebox.askyesno("Confirm Reset", 
                              "Are you sure you want to reset all settings to defaults?"):
            # Reset config values
            Config.THRESHOLD = 0.7
            Config.SAMPLE_RATE = 16000
            Config.AUDIO_DURATION = 3.0
            
            # Reset notification settings but keep the model path
            Config.PARENT_PHONE = None
            Config.PARENT_EMAIL = None
            
            # Save config
            Config.save()
            
            # Reload settings in UI
            self.load_settings()
            
            # Call callback if registered
            if self.callback:
                self.callback()
            
            messagebox.showinfo("Success", "Settings reset to defaults")
    
    def update_threshold_label(self, event=None):
        """Update threshold value label when slider moves"""
        self.threshold_value_label.config(text=f"{self.threshold_var.get():.2f}")
    
    def browse_model(self):
        """Browse for model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def test_notification(self):
        """Send test notification"""
        # Update notification system with current settings
        self.notification.parent_phone = self.phone_var.get() or None
        self.notification.parent_email = self.email_var.get() or None
        
        # Send test notification
        success = self.notification.send_test_notification()
        
        if success:
            messagebox.showinfo("Success", "Test notification sent successfully")
        else:
            messagebox.showwarning("Warning", 
                                  "Failed to send test notification. Please check settings.")