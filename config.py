"""
Configuration settings for the Scream Detection System
"""

import os
import json
from pathlib import Path

class Config:
    # Default paths
    BASE_DIR = Path(__file__).parent.absolute()
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "saved", "screaming_detection_model.h5")
    DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data")
    CONFIG_FILE = os.path.join(os.path.dirname(BASE_DIR), "config.json")
    
    # Audio settings
    SAMPLE_RATE = 16000
    AUDIO_DURATION = 3  # in seconds
    CHUNK_SIZE = 1024
    
    # Model settings - UPDATED: Lower threshold for better sensitivity
    THRESHOLD = 0.45  # Changed from 0.7 to 0.45 for better sensitivity
    
    # Notification settings
    PARENT_PHONE = None
    PARENT_EMAIL = None
    
    @classmethod
    def initialize(cls):
        """Initialize configuration and create necessary directories"""
        # Create necessary directories
        os.makedirs(os.path.join(os.path.dirname(cls.BASE_DIR), "models", "saved"), exist_ok=True)
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(cls.BASE_DIR), "clips"), exist_ok=True)
        
        # Load configuration
        cls.load()
    
    @classmethod
    def load(cls):
        """Load configuration from file"""
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
                
                print(f"Configuration loaded from {cls.CONFIG_FILE}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
    
    @classmethod
    def save(cls):
        """Save configuration to file"""
        config_data = {
            "MODEL_PATH": cls.MODEL_PATH,
            "DATA_PATH": cls.DATA_PATH,
            "SAMPLE_RATE": cls.SAMPLE_RATE,
            "AUDIO_DURATION": cls.AUDIO_DURATION,
            "THRESHOLD": cls.THRESHOLD,
            "PARENT_PHONE": cls.PARENT_PHONE,
            "PARENT_EMAIL": cls.PARENT_EMAIL
        }
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(cls.CONFIG_FILE), exist_ok=True)
            
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            print(f"Configuration saved to {cls.CONFIG_FILE}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

# Initialize configuration
Config.initialize()