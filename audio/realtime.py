"""
Real-time audio detection module for the Scream Detection System
"""

import time
import threading
import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from datetime import datetime
from .processor import AudioProcessor
from .processor import  ScreamDetector
from ..utils.notification import NotificationSystem
from ..config import Config

class RealtimeScreamDetector:
    def __init__(self, model_path=None, device_index=None, callback=None):
        """
        Initialize real-time scream detector
        
        Args:
            model_path (str, optional): Path to the model file
            device_index (int, optional): Audio device index
            callback (function, optional): Callback for detection updates
        """
        self.processor = AudioProcessor()
        self.detector = ScreamDetector(model_path)
        self.device_index = device_index
        self.callback = callback
        self.notification = NotificationSystem()
        
        # Audio parameters
        self.sample_rate = Config.SAMPLE_RATE
        self.chunk_size = Config.CHUNK_SIZE
        self.channels = 1
        
        # Monitoring state
        self.running = False
        self.stream = None
        self.audio_queue = Queue()
        
        # Auto-detection parameters
        self.auto_detect = True
        self.detect_interval = 0.5  # seconds between detections (faster updates)
        self.last_detection_time = 0
        self.audio_buffer = np.zeros(int(self.sample_rate * Config.AUDIO_DURATION), dtype=np.float32)
        # Detection history (to prevent too many consecutive detections)
        self.detection_cooldown = 3.0  # seconds to wait after a detection (reduced for better response)
        self.last_scream_time = 0
        
        # Last detection results
        self.last_detection = None
        
        # Print detector configuration for debugging
        print(f"RealtimeScreamDetector initialized with model: {model_path}")
        print(f"Using threshold: {self.detector.threshold if self.detector else 'None'}")
        print(f"Device index: {device_index}")
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback for sounddevice stream
        
        This is called for each audio block captured by sounddevice
        """
        if status:
            print(f"Audio callback status: {status}")
            
        # Get the audio data and put it in the queue
        audio_data = indata[:, 0]  # Get the first channel
        self.audio_queue.put(audio_data.copy())
        
        # Also update the rolling buffer directly
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data):] = audio_data
        
        # Auto-detect if enabled
        if self.auto_detect:
            current_time = time.time()
            if current_time - self.last_detection_time >= self.detect_interval:
                self.last_detection_time = current_time
                
                # Start detection in a separate thread to avoid blocking the audio callback
                threading.Thread(target=self._auto_detect, daemon=True).start()
    
    def _auto_detect(self):
        """
        Perform scream detection on the current audio buffer
        """
        try:
            # Create a copy of the buffer to avoid race conditions
            buffer_copy = self.audio_buffer.copy()
            
            # Calculate audio energy for activity detection
            energy = np.sum(buffer_copy**2) / len(buffer_copy)
            
            # Only process if there's significant audio (reduces false detections on silence)
            if energy > 0.0001:  # Threshold for audio activity
                # Process the audio
                is_screaming, confidence, mel_spec_db = self.detector.detect(buffer_copy)
                
                # Print detection results for debugging
                print(f"Detection result: {is_screaming}, Confidence: {confidence:.4f}, Energy: {energy:.6f}")
                
                # Update detection results
                self.last_detection = (is_screaming, confidence, mel_spec_db)
                
                # Call callback if provided
                if self.callback:
                    self.callback(is_screaming, confidence, mel_spec_db)
                
                # Handle scream detection
                current_time = time.time()
                if is_screaming and (current_time - self.last_scream_time) > self.detection_cooldown:
                    self.last_scream_time = current_time
                    
                    # Save audio clip for verification
                    clip_path = self.processor.save_audio_clip(buffer_copy, prefix="scream_detected")
                    
                    # Send notification
                    if clip_path:
                        self.notification.notify_emergency_contacts(
                            confidence, 
                            audio_path=clip_path
                        )
            else:
                # If no significant audio, still update the UI but with low confidence
                if self.callback:
                    # Create empty mel spectrogram for visualization
                    empty_mel = np.zeros((128, 256))
                    self.callback(False, 0.0, empty_mel)
        except Exception as e:
            print(f"Error in auto detection: {e}")
            import traceback
            traceback.print_exc()
    
    def start_monitoring(self):
        """
        Start real-time audio monitoring
        
        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        if self.running:
            print("Monitoring is already running")
            return False
            
        try:
            # Make sure sounddevice is available
            import sounddevice as sd
            
            # Set up stream
            self.running = True
            
            # Create stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device_index,
                channels=self.channels,
                dtype='float32',
                callback=self.audio_callback
            )
            
            # Start stream
            self.stream.start()
            
            print(f"Monitoring started on device {self.device_index if self.device_index is not None else 'default'}")
            print(f"Using detection threshold: {self.detector.threshold}")
            return True
            
        except Exception as e:
            self.running = False
            print(f"Error starting audio monitoring: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_monitoring(self):
        """
        Stop real-time audio monitoring
        """
        if not self.running:
            return
            
        self.running = False
        
        # Stop and close stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
            finally:
                self.stream = None
        
        print("Monitoring stopped")
    
    @staticmethod
    def get_available_devices():
        """
        Get a list of available audio input devices
        
        Returns:
            list: List of (index, name) tuples for available input devices
        """
        try:
            import sounddevice as sd
            devices = []
            
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:  # Input device
                    devices.append((i, device['name']))
            
            return devices
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            return []