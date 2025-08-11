"""
Real-time monitoring tab for the Scream Detection System GUI
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import numpy as np

from ..audio.realtime import RealtimeScreamDetector
from ..utils.visualization import AudioVisualizer
from ..config import Config

class RealtimeMonitorTab(ttk.Frame):
    def __init__(self, parent):
        """
        Initialize real-time monitoring tab
        
        Args:
            parent (ttk.Notebook): Parent notebook widget
        """
        super().__init__(parent, padding=10)
        
        # Initialize components
        self.model_path = Config.MODEL_PATH
        self.detector = None
        self.monitoring_thread = None
        self.device_index = None
        
        # Auto-detection settings
        self.auto_detect_enabled = True
        
        # Create UI components
        self.create_widgets()
        
        # Refresh audio devices
        self.refresh_devices()
    
    def create_widgets(self):
        """Create tab widgets"""
        # Main container with scrollbar
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Control frame (top)
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Device selection
        device_label = ttk.Label(control_frame, text="Audio Input Device:")
        device_label.pack(side=tk.LEFT, padx=5)
        
        self.device_var = tk.StringVar()
        self.device_dropdown = ttk.Combobox(control_frame, textvariable=self.device_var, state="readonly", width=30)
        self.device_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        refresh_btn = ttk.Button(control_frame, text="Refresh Devices", command=self.refresh_devices)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Monitoring control buttons
        self.start_btn = ttk.Button(
            control_frame, 
            text="Start Monitoring", 
            command=self.start_monitoring, 
            style="Accent.TButton",
            width=20  # Make button wider for better visibility
        )
        self.start_btn.pack(side=tk.RIGHT, padx=10)  # Increased padding for better visibility
        
        self.stop_btn = ttk.Button(
            control_frame, 
            text="Stop Monitoring", 
            command=self.stop_monitoring, 
            state=tk.DISABLED,
            width=20  # Make button wider for better visibility
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=5)
        
        # Threshold adjustment
        threshold_frame = ttk.Frame(main_container)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        threshold_label = ttk.Label(threshold_frame, text="Detection Threshold:")
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=Config.THRESHOLD)
        threshold_slider = ttk.Scale(
            threshold_frame, 
            from_=0.05, to=0.95, 
            variable=self.threshold_var, 
            command=self.update_threshold
        )
        threshold_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.threshold_value_label = ttk.Label(threshold_frame, text=f"{Config.THRESHOLD:.2f}")
        self.threshold_value_label.pack(side=tk.LEFT, padx=5)
        
        # Auto-detection checkbox
        self.auto_detect_var = tk.BooleanVar(value=True)
        auto_detect_check = ttk.Checkbutton(
            main_container, 
            text="Enable automatic detection", 
            variable=self.auto_detect_var,
            command=self.toggle_auto_detect
        )
        auto_detect_check.pack(pady=5, anchor=tk.W)
        
        # Status indicators frame
        status_frame = ttk.Frame(main_container)
        status_frame.pack(fill=tk.X, pady=10)
        
        # Monitoring status with larger font and bold
        status_label = ttk.Label(status_frame, text="Status:", font=("Arial", 11))
        status_label.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Not Monitoring")
        self.status_indicator = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            font=("Arial", 14, "bold")
        )
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        
        # Detection indicator (visual) - larger size
        self.indicator_canvas = tk.Canvas(
            status_frame, 
            width=30, 
            height=30, 
            highlightthickness=1,
            highlightbackground="black",
            bg="#f0f0f0"
        )
        self.indicator_canvas.pack(side=tk.LEFT, padx=15)
        self.indicator = self.indicator_canvas.create_oval(3, 3, 27, 27, fill="gray")
        
        # Current detection status (large text)
        detection_frame = ttk.Frame(main_container)
        detection_frame.pack(fill=tk.X, pady=10)
        
        self.detection_var = tk.StringVar(value="No Detection")
        detection_label = ttk.Label(
            detection_frame, 
            textvariable=self.detection_var,
            font=("Arial", 24, "bold"),
            foreground="black"
        )
        detection_label.pack(pady=10, fill=tk.X)
        
        # Real-time visualization frame
        viz_frame = ttk.Frame(main_container)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create visualization figures
        self.mel_frame = ttk.LabelFrame(viz_frame, text="Real-time Mel Spectrogram")
        self.mel_frame.pack(fill=tk.BOTH, expand=True)
        
        self.mel_figure = Figure(figsize=(8, 4), dpi=100)
        self.mel_canvas = FigureCanvasTkAgg(self.mel_figure, self.mel_frame)
        self.mel_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create mel spectrogram visualizer
        self.mel_viz = AudioVisualizer(self.mel_figure, self.mel_canvas)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_container, text="Detection Results")
        results_frame.pack(fill=tk.X, pady=10)
        
        # Real-time confidence
        confidence_frame = ttk.Frame(results_frame)
        confidence_frame.pack(fill=tk.X, padx=20, pady=10)
        
        confidence_label = ttk.Label(confidence_frame, text="Confidence:", font=("Arial", 11))
        confidence_label.pack(side=tk.LEFT, padx=5)
        
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(
            confidence_frame, 
            orient=tk.HORIZONTAL, 
            length=300, 
            mode='determinate', 
            variable=self.confidence_var
        )
        self.confidence_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.confidence_text = ttk.Label(confidence_frame, text="0%", font=("Arial", 11, "bold"))
        self.confidence_text.pack(side=tk.LEFT, padx=5)
        
        # Detection history
        history_frame = ttk.Frame(results_frame)
        history_frame.pack(fill=tk.X, padx=20, pady=10)
        
        history_label = ttk.Label(history_frame, text="Detection History:", font=("Arial", 11))
        history_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Create scrolled text widget for history
        self.history_text = tk.Text(
            history_frame, 
            height=5, 
            width=50, 
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        self.history_text.pack(fill=tk.X, expand=True, side=tk.LEFT)
        
        history_scrollbar = ttk.Scrollbar(history_frame, command=self.history_text.yview)
        history_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.history_text.config(yscrollcommand=history_scrollbar.set)
        
        # Make the text widget read-only
        self.history_text.config(state=tk.DISABLED)
    
    def update_threshold(self, event=None):
        """Update threshold value and label"""
        value = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{value:.2f}")
        
        # Update detector threshold if running
        if self.detector:
            self.detector.detector.threshold = value
            self.add_history_entry(f"Detection threshold updated to {value:.2f}")
    
    def refresh_devices(self):
        """Refresh audio device list"""
        try:
            # Get all available devices
            devices = RealtimeScreamDetector.get_available_devices()
            
            if devices:
                # Format devices for display
                device_options = [f"{name} (ID: {idx})" for idx, name in devices]
                self.device_dropdown['values'] = device_options
                
                # Set first device as default if none selected
                if not self.device_var.get() and device_options:
                    self.device_dropdown.current(0)
                    # Extract device index from selected item
                    self.device_index = devices[0][0]
            else:
                self.device_dropdown['values'] = ["No input devices found"]
                self.device_dropdown.current(0)
                self.device_index = None
                
        except Exception as e:
            self.device_dropdown['values'] = [f"Error: {str(e)}"]
            self.device_dropdown.current(0)
            self.device_index = None
    
    def set_model_path(self, model_path):
        """
        Set model path
        
        Args:
            model_path (str): Path to model file
        """
        self.model_path = model_path
        
        # If monitoring is active, restart it
        if self.detector and self.detector.running:
            self.stop_monitoring()
            self.start_monitoring()
    
    def set_threshold(self, threshold):
        """
        Set detection threshold
        
        Args:
            threshold (float): Detection threshold (0-1)
        """
        self.threshold_var.set(threshold)
        self.update_threshold()
        if self.detector:
            self.detector.detector.threshold = threshold
    
    def toggle_auto_detect(self):
        """Toggle auto-detection on/off"""
        if self.detector:
            self.detector.auto_detect = self.auto_detect_var.get()
            
            if self.auto_detect_var.get():
                self.add_history_entry("Auto-detection enabled")
            else:
                self.add_history_entry("Auto-detection disabled")
    
    def start_monitoring(self):
        """Start real-time audio monitoring"""
        if not self.model_path or not os.path.exists(self.model_path):
            messagebox.showwarning("Warning", "No model available. Please train or import a model first.")
            return
        
        # Get selected device index
        selected = self.device_var.get()
        if "ID:" in selected:
            # Extract device index from dropdown text
            try:
                self.device_index = int(selected.split("ID: ")[1].rstrip(")"))
            except (ValueError, IndexError):
                # Fallback to default device
                self.device_index = None
        
        try:
            # Create detector
            self.detector = RealtimeScreamDetector(
                model_path=self.model_path,
                device_index=self.device_index,
                callback=self.update_detection
            )
            
            # Set auto-detect from UI
            self.detector.auto_detect = self.auto_detect_var.get()
            
            # Set threshold from UI
            if hasattr(self.detector, 'detector') and self.detector.detector:
                self.detector.detector.threshold = self.threshold_var.get()
            
            # Start monitoring
            success = self.detector.start_monitoring()
            
            if success:
                # Update UI
                self.status_var.set("Monitoring Active")
                self.indicator_canvas.itemconfig(self.indicator, fill="green")
                self.detection_var.set("Listening...")
                
                # Enable/disable buttons
                self.start_btn.configure(state=tk.DISABLED)
                self.stop_btn.configure(state=tk.NORMAL)
                self.device_dropdown.configure(state=tk.DISABLED)
                
                # Add to history
                self.add_history_entry(f"Monitoring started on device {self.device_index if self.device_index is not None else 'default'}")
                self.add_history_entry(f"Using detection threshold: {self.threshold_var.get():.2f}")
            else:
                messagebox.showerror("Error", "Failed to start monitoring. Check audio device.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop real-time audio monitoring"""
        if self.detector:
            self.detector.stop_monitoring()
            self.detector = None
            
            # Update UI
            self.status_var.set("Not Monitoring")
            self.indicator_canvas.itemconfig(self.indicator, fill="gray")
            self.confidence_var.set(0)
            self.confidence_text.config(text="0%")
            self.detection_var.set("No Detection")
            
            # Clear visualization
            self.mel_figure.clear()
            self.mel_canvas.draw()
            
            # Enable/disable buttons
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.device_dropdown.configure(state="readonly")
            
            # Add to history
            self.add_history_entry("Monitoring stopped")
    
    def update_detection(self, is_screaming, confidence, mel_spec_db):
        """
        Update UI with detection results
        
        Args:
            is_screaming (bool): True if scream detected
            confidence (float): Detection confidence (0-1)
            mel_spec_db (numpy.ndarray): Mel spectrogram for visualization
        """
        # This function is called from a different thread
        # Use after() to update UI from the main thread
        self.after(10, lambda: self._update_ui(is_screaming, confidence, mel_spec_db))
    
    def _update_ui(self, is_screaming, confidence, mel_spec_db):
        """Update UI elements from the main thread"""
        # Update confidence display
        self.confidence_var.set(confidence * 100)
        self.confidence_text.config(text=f"{confidence*100:.1f}%")
        
        # Update indicator color and detection status
        if is_screaming:
            # Red alert for scream detected
            self.indicator_canvas.itemconfig(self.indicator, fill="red")
            # Large text showing SCREAM DETECTED
            self.detection_var.set("⚠️ SCREAM DETECTED! ⚠️")
            # Use red text for alarm status
            self.status_indicator.configure(foreground="red")
            # Add history entry
            self.add_history_entry(f"⚠️ SCREAM DETECTED with {confidence*100:.1f}% confidence")
            # Flash the indicator for attention
            self.flash_indicator(5)  # Flash 5 times
        else:
            # Green for normal monitoring
            self.indicator_canvas.itemconfig(self.indicator, fill="green")
            # Normal text
            self.detection_var.set("No Scream Detected")
            # Use normal text color
            self.status_indicator.configure(foreground="black")
        
        # Update mel spectrogram visualization
        if mel_spec_db is not None:
            self.mel_viz.plot_melspectrogram(
                mel_spec_db,
                sample_rate=Config.SAMPLE_RATE,
                title="Real-time Mel Spectrogram (dB)"
            )
    
    def flash_indicator(self, count=3):
        """Flash the indicator for attention"""
        if count <= 0:
            return
            
        # Get current color
        current_color = self.indicator_canvas.itemcget(self.indicator, "fill")
        
        # Toggle color
        new_color = "yellow" if current_color == "red" else "red"
        
        # Change color
        self.indicator_canvas.itemconfig(self.indicator, fill=new_color)
        
        # Schedule next flash
        self.after(300, lambda: self.flash_indicator(count - 1))
    
    def add_history_entry(self, text):
        """
        Add entry to history log
        
        Args:
            text (str): Text to add
        """
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        entry = f"[{timestamp}] {text}\n"
        
        # Update text widget
        self.history_text.config(state=tk.NORMAL)
        self.history_text.insert(tk.END, entry)
        self.history_text.see(tk.END)  # Scroll to bottom
        self.history_text.config(state=tk.DISABLED)