"""
File analysis tab for the Scream Detection System GUI
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from ..audio.processor import ScreamDetector, AudioProcessor
from ..utils.visualization import AudioVisualizer
from ..utils.notification import NotificationSystem
from ..config import Config

class FileAnalysisTab(ttk.Frame):
    def __init__(self, parent):
        """
        Initialize file analysis tab
        
        Args:
            parent (ttk.Notebook): Parent notebook widget
        """
        super().__init__(parent, padding=10)
        
        # Initialize components
        self.model_path = Config.MODEL_PATH
        self.detector = None
        self.audio_processor = AudioProcessor()
        self.notification = NotificationSystem()
        
        # Current audio data
        self.current_audio = None
        self.current_mel_spec = None
        self.last_confidence = 0.0
        
        # Create UI components
        self.create_widgets()
        
        # Try to initialize detector
        self.initialize_detector()
    
    def create_widgets(self):
        """Create tab widgets"""
        # Create a main container with scrollbar for better display on small screens
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top controls frame for file upload and analysis buttons
        top_controls = ttk.Frame(main_container)
        top_controls.pack(fill=tk.X, pady=5)
        
        # File upload button
        upload_btn = ttk.Button(top_controls, text="Upload Audio File", command=self.upload_file)
        upload_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Analyze button - MOVED UP for visibility
        analyze_btn = ttk.Button(top_controls, text="Analyze", 
                                command=self.analyze_file, 
                                style="Accent.TButton")  # Use accent style for visibility
        analyze_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Play button
        play_btn = ttk.Button(top_controls, text="Play Audio", command=self.play_audio)
        play_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Debug buttons
        debug_btn = ttk.Button(top_controls, text="Debug", command=self.test_direct_feature)
        debug_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        test_threshold_btn = ttk.Button(top_controls, text="Test Thresholds", command=self.test_thresholds)
        test_threshold_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # File path display
        path_frame = ttk.Frame(main_container)
        path_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_path_label = ttk.Label(path_frame, textvariable=self.file_path_var, wraplength=600)
        file_path_label.pack(pady=5)
        
        # Threshold adjustment
        threshold_frame = ttk.Frame(main_container)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        threshold_label = ttk.Label(threshold_frame, text="Detection Threshold:")
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=Config.THRESHOLD)
        threshold_slider = ttk.Scale(threshold_frame, from_=0.05, to=0.95, 
                                   variable=self.threshold_var, 
                                   command=self.update_threshold_value)
        threshold_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.threshold_value_label = ttk.Label(threshold_frame, text=f"{Config.THRESHOLD:.2f}")
        self.threshold_value_label.pack(side=tk.LEFT, padx=5)
        
        # Visualization frame
        viz_frame = ttk.Frame(main_container)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Audio waveform
        waveform_frame = ttk.LabelFrame(viz_frame, text="Audio Waveform")
        waveform_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.waveform_figure = Figure(figsize=(6, 3), dpi=100)
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_figure, waveform_frame)
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mel spectrogram
        mel_frame = ttk.LabelFrame(viz_frame, text="Mel Spectrogram (dB)")
        mel_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.mel_figure = Figure(figsize=(6, 3), dpi=100)
        self.mel_canvas = FigureCanvasTkAgg(self.mel_figure, mel_frame)
        self.mel_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create visualizers
        self.waveform_viz = AudioVisualizer(self.waveform_figure, self.waveform_canvas)
        self.mel_viz = AudioVisualizer(self.mel_figure, self.mel_canvas)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_container, text="Analysis Results")
        results_frame.pack(fill=tk.X, pady=5)
        
        # Detection result label
        self.detection_var = tk.StringVar()
        self.detection_var.set("Upload an audio file to analyze")
        detection_label = ttk.Label(results_frame, textvariable=self.detection_var, font=("Arial", 14))
        detection_label.pack(pady=5)
        
        # Confidence progress bar
        confidence_frame = ttk.Frame(results_frame)
        confidence_frame.pack(fill=tk.X, padx=20, pady=5)
        
        confidence_label = ttk.Label(confidence_frame, text="Confidence:")
        confidence_label.pack(side=tk.LEFT, padx=5)
        
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(confidence_frame, orient=tk.HORIZONTAL, 
                                             length=300, mode='determinate', 
                                             variable=self.confidence_var)
        self.confidence_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.confidence_text = ttk.Label(confidence_frame, text="0%")
        self.confidence_text.pack(side=tk.LEFT, padx=5)
        
        # Bottom controls frame - for notification settings
        bottom_controls = ttk.Frame(main_container)
        bottom_controls.pack(fill=tk.X, pady=5)
        
        # Notification checkbox
        self.notify_var = tk.BooleanVar(value=False)
        notify_check = ttk.Checkbutton(bottom_controls, text="Send notification if scream detected", 
                                     variable=self.notify_var)
        notify_check.pack(side=tk.LEFT, padx=5)
        
        # Test notification button
        test_notify_btn = ttk.Button(bottom_controls, text="Test Notification", 
                                   command=self.send_test_notification)
        test_notify_btn.pack(side=tk.RIGHT, padx=5)
        
        # Debug info
        self.debug_var = tk.StringVar()
        debug_label = ttk.Label(main_container, textvariable=self.debug_var, 
                              wraplength=800, justify=tk.LEFT, font=("Courier", 9))
        debug_label.pack(fill=tk.X, pady=5)
    
    def update_threshold_value(self, event=None):
        """Update threshold value label and detector threshold"""
        value = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{value:.2f}")
        
        if self.detector:
            self.detector.threshold = value
            print(f"Threshold updated to: {value}")
    
    def initialize_detector(self):
        """Initialize the scream detector"""
        if os.path.exists(self.model_path):
            try:
                self.detector = ScreamDetector(self.model_path)
                # Set threshold from slider
                self.detector.threshold = self.threshold_var.get()
                print(f"Detector initialized with model: {self.model_path}")
                print(f"Using threshold: {self.detector.threshold}")
                self.debug_var.set(f"Model loaded: {self.model_path}\nThreshold: {self.detector.threshold}\nInput shape: {self.detector.model.input_shape if self.detector.model else 'Unknown'}")
                return True
            except Exception as e:
                print(f"Failed to initialize detector: {e}")
                self.debug_var.set(f"Failed to initialize detector: {e}")
                return False
        else:
            print(f"Model not found at: {self.model_path}")
            self.debug_var.set(f"Model not found at: {self.model_path}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            return False
    
    def set_model_path(self, model_path):
        """
        Set model path and reinitialize detector
        
        Args:
            model_path (str): Path to model file
        """
        self.model_path = model_path
        self.initialize_detector()
    
    def set_threshold(self, threshold):
        """
        Set detection threshold
        
        Args:
            threshold (float): Detection threshold (0-1)
        """
        self.threshold_var.set(threshold)
        self.update_threshold_value()
        if self.detector:
            self.detector.threshold = threshold
    
    def upload_file(self):
        """Handle file upload button click"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav;*.mp3"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            
            # Reset current data
            self.current_audio = None
            self.current_mel_spec = None
            self.detection_var.set("Click 'Analyze' to process the audio")
            self.confidence_var.set(0)
            self.confidence_text.config(text="0%")
            
            # Clear visualizations
            self.waveform_figure.clear()
            self.mel_figure.clear()
            self.waveform_canvas.draw()
            self.mel_canvas.draw()
    
    def analyze_file(self):
        """Handle analyze button click"""
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showwarning("Warning", "Please upload an audio file first")
            return
        
        if not self.detector:
            if not self.initialize_detector():
                messagebox.showwarning("Warning", 
                                      "No model available. Please train or import a model first.")
                return
        
        try:
            # Update UI to show processing
            self.detection_var.set("Processing...")
            self.confidence_var.set(0)
            self.confidence_text.config(text="0%")
            self.update()  # Force UI update
            
            # Detect screams in the file
            is_screaming, confidence, mel_spec_db = self.detector.detect(filepath=file_path)
            
            # Load audio for visualization
            self.current_audio = self.audio_processor.load_audio(file_path)
            self.current_mel_spec = mel_spec_db
            
            # Update visualizations
            self.update_visualizations()
            
            # Update detection result
            if is_screaming:
                self.detection_var.set("⚠️ SCREAM DETECTED!")
                
                # Send notification if checkbox is checked
                if self.notify_var.get():
                    self.notification.notify_emergency_contacts(
                        confidence,
                        audio_path=file_path
                    )
            else:
                self.detection_var.set("No scream detected")
            
            # Update confidence
            self.confidence_var.set(confidence * 100)
            self.confidence_text.config(text=f"{confidence*100:.1f}%")
            
            # Store the last confidence value for testing
            self.last_confidence = confidence
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to analyze audio: {str(e)}")
            self.detection_var.set("Analysis failed")
    
    def update_visualizations(self):
        """Update visualizations with current audio data"""
        if self.current_audio is not None:
            self.waveform_viz.plot_waveform(
                self.current_audio, 
                sample_rate=Config.SAMPLE_RATE, 
                title="Audio Waveform"
            )
        
        if self.current_mel_spec is not None:
            self.mel_viz.plot_melspectrogram(
                self.current_mel_spec,
                sample_rate=Config.SAMPLE_RATE,
                title="Mel Spectrogram (dB)"
            )
    
    def play_audio(self):
        """Play the current audio file"""
        file_path = self.file_path_var.get()
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showinfo("Info", "No audio file selected")
            return
        
        try:
            # Use system's default audio player
            import platform
            import subprocess
            
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', file_path))
            elif platform.system() == 'Windows':
                os.startfile(file_path)
            else:  # Linux
                subprocess.call(('xdg-open', file_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {str(e)}")
    
    def send_test_notification(self):
        """Send a test notification"""
        success = self.notification.send_test_notification()
        
        if success:
            messagebox.showinfo("Success", "Test notification sent successfully")
        else:
            messagebox.showwarning("Warning", "Failed to send test notification")
    
    def test_direct_feature(self):
        """Test direct feature extraction and prediction on current file"""
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showwarning("Warning", "Please upload an audio file first")
            return
        
        try:
            # Load and process audio
            print(f"Testing direct feature extraction on {file_path}")
            audio_data = self.audio_processor.load_audio(file_path)
            mel_spec_model, mel_spec_db = self.audio_processor.extract_features(audio_data)
            
            # Print feature shape
            feature_info = f"Feature shape: {mel_spec_model.shape}"
            print(feature_info)
            
            # Try direct prediction if model exists
            if self.detector and self.detector.model:
                print("Making direct prediction with model")
                raw_prediction = self.detector.model.predict(mel_spec_model, verbose=1)
                
                prediction_info = f"Raw prediction: {raw_prediction}"
                print(prediction_info)
                
                # Try different interpretation of outputs
                interpretation = ""
                if len(raw_prediction.shape) > 1 and raw_prediction.shape[1] == 2:
                    not_scream_prob = raw_prediction[0][0]
                    scream_prob = raw_prediction[0][1]
                    interpretation = f"As binary: Not scream: {not_scream_prob:.4f}, Scream: {scream_prob:.4f}"
                    
                    # Test different thresholds
                    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    threshold_results = "\nThreshold tests:\n"
                    for t in thresholds:
                        result = "DETECTED" if scream_prob > t else "not detected"
                        threshold_results += f"  {t:.1f}: {result}\n"
                else:
                    scream_prob = raw_prediction[0][0]
                    interpretation = f"As single: {scream_prob:.4f}"
                    
                    # Test different thresholds
                    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    threshold_results = "\nThreshold tests:\n"
                    for t in thresholds:
                        result = "DETECTED" if scream_prob > t else "not detected"
                        threshold_results += f"  {t:.1f}: {result}\n"
                
                print(interpretation)
                print(threshold_results)
                
                # Update debug info
                self.debug_var.set(f"{feature_info}\n{prediction_info}\n{interpretation}{threshold_results}")
                
                # Also update the analysis results with this information
                self.detection_var.set(f"Debug: {'SCREAM' if scream_prob > self.detector.threshold else 'No scream'}")
                self.confidence_var.set(scream_prob * 100)
                self.confidence_text.config(text=f"{scream_prob*100:.1f}%")
                
                # Store for threshold testing
                self.last_confidence = scream_prob
            else:
                self.debug_var.set(f"{feature_info}\nNo model available for prediction.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.debug_var.set(f"Feature test error: {str(e)}")
    
    def test_thresholds(self):
        """Test different thresholds on the last prediction"""
        if not hasattr(self, 'last_confidence') or self.last_confidence == 0:
            messagebox.showinfo("Info", "Please analyze an audio file first")
            return
        
        # Create a new window for threshold testing
        test_window = tk.Toplevel(self)
        test_window.title("Threshold Testing")
        test_window.geometry("400x400")
        
        # Create a frame for the threshold testing
        frame = ttk.Frame(test_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Show current confidence
        confidence_label = ttk.Label(frame, text=f"Current confidence: {self.last_confidence:.4f}")
        confidence_label.pack(pady=10)
        
        # Create a list of thresholds to test
        thresholds = np.linspace(0.05, 0.95, 19)  # 19 values from 0.05 to 0.95
        
        # Create a frame for the threshold results
        results_frame = ttk.Frame(frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas for vertical scrolling
        canvas = tk.Canvas(results_frame)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add threshold test results
        for threshold in thresholds:
            result = "DETECTED" if self.last_confidence > threshold else "not detected"
            color = "#e6ffe6" if result == "DETECTED" else "#ffe6e6"  # Light green or light red
            
            # Create a frame for each threshold
            threshold_frame = ttk.Frame(scrollable_frame)
            threshold_frame.pack(fill=tk.X, pady=2)
            
            # Create a label for the threshold
            threshold_label = ttk.Label(threshold_frame, text=f"Threshold {threshold:.2f}:")
            threshold_label.pack(side=tk.LEFT, padx=5)
            
            # Create a label for the result
            result_label = ttk.Label(threshold_frame, text=result)
            result_label.pack(side=tk.LEFT, padx=5)
            
            # If this is close to the current threshold, highlight it
            if abs(threshold - self.threshold_var.get()) < 0.05:
                threshold_frame.configure(style="Highlight.TFrame")
                threshold_label.configure(text=f"Current Threshold {threshold:.2f}:")
        
        # Create a button to apply a selected threshold
        apply_frame = ttk.Frame(frame)
        apply_frame.pack(fill=tk.X, pady=10)
        
        apply_label = ttk.Label(apply_frame, text="Set new threshold:")
        apply_label.pack(side=tk.LEFT, padx=5)
        
        new_threshold_var = tk.DoubleVar(value=self.threshold_var.get())
        new_threshold_entry = ttk.Entry(apply_frame, textvariable=new_threshold_var, width=10)
        new_threshold_entry.pack(side=tk.LEFT, padx=5)
        
        apply_btn = ttk.Button(apply_frame, text="Apply", 
                             command=lambda: self.apply_new_threshold(new_threshold_var.get(), test_window))
        apply_btn.pack(side=tk.LEFT, padx=5)
    
    def apply_new_threshold(self, new_threshold, window=None):
        """Apply a new threshold value"""
        # Validate threshold
        if new_threshold < 0 or new_threshold > 1:
            messagebox.showerror("Error", "Threshold must be between 0 and 1")
            return
            
        # Update threshold
        self.threshold_var.set(new_threshold)
        self.update_threshold_value()
        
        # Re-analyze the current file with the new threshold
        if hasattr(self, 'last_confidence'):
            is_screaming = self.last_confidence > new_threshold
            result = "SCREAM DETECTED!" if is_screaming else "No scream detected"
            self.detection_var.set(f"⚠️ {result}" if is_screaming else result)
        
        # Close the threshold testing window if provided
        if window:
            window.destroy()