"""
Audio processing utilities for the Scream Detection System
"""

import os
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
from tensorflow.keras.models import load_model
from ..config import Config

class AudioProcessor:
    def __init__(self, sample_rate=None, duration=None):
        """
        Initialize the audio processor
        
        Args:
            sample_rate (int, optional): Sample rate for audio processing
            duration (float, optional): Duration of audio clips
        """
        self.sr = sample_rate or Config.SAMPLE_RATE
        self.duration = duration or Config.AUDIO_DURATION
    
    def load_audio(self, filepath):
        """
        Load an audio file
        
        Args:
            filepath (str): Path to the audio file
            
        Returns:
            numpy.ndarray: Processed audio data
        """
        try:
            y, sr = librosa.load(filepath, sr=self.sr, duration=self.duration)
            return self.preprocess_audio(y)
        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}")
            # Return a silent audio buffer instead of failing
            return np.zeros(int(self.sr * self.duration), dtype=np.float32)
    
    def preprocess_audio(self, audio_data):
        """
        Preprocess audio data
        
        Args:
            audio_data (numpy.ndarray): Raw audio data
            
        Returns:
            numpy.ndarray: Processed audio data
        """
        # Make sure audio data is the right type
        audio_data = np.asarray(audio_data, dtype=np.float32)
        
        # Handle NaN and Inf values
        audio_data = np.nan_to_num(audio_data)
        
        # Normalize audio to [-1, 1] range if it's not already
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # Pad or trim audio to fixed length
        if len(audio_data) < int(self.sr * self.duration):
            audio_data = np.pad(audio_data, (0, int(self.sr * self.duration) - len(audio_data)), 'constant')
        else:
            audio_data = audio_data[:int(self.sr * self.duration)]
            
        return audio_data
    
    def extract_features(self, audio_data):
        """
        Extract mel spectrogram features from audio data
        
        Args:
            audio_data (numpy.ndarray): Audio data
            
        Returns:
            tuple: (mel_spectrogram, mel_spectrogram_for_display)
        """
        # Extract Mel Spectrogram with window and hop length matching training
        hop_length = 512  # Standard hop length
        n_fft = 2048      # Standard n_fft size
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.sr, 
            n_mels=128,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create copy for model input (reshaped)
        mel_spec_model = mel_spec_db.copy()
        mel_spec_model = mel_spec_model.reshape(1, mel_spec_model.shape[0], mel_spec_model.shape[1], 1)
        
        return mel_spec_model, mel_spec_db
    
    def save_audio_clip(self, audio_data, prefix="audio_clip"):
        """
        Save audio data to a file
        
        Args:
            audio_data (numpy.ndarray): Audio data to save
            prefix (str, optional): Filename prefix
            
        Returns:
            str: Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        
        # Create 'clips' directory if it doesn't exist
        clips_dir = os.path.join(os.path.dirname(Config.BASE_DIR), "clips")
        os.makedirs(clips_dir, exist_ok=True)
        
        filepath = os.path.join(clips_dir, filename)
        
        # Ensure audio data is in the correct format for saving
        audio_data = np.asarray(audio_data, dtype=np.float32)
        audio_data = np.nan_to_num(audio_data, nan=0.0)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Write the file with error handling
        try:
            sf.write(filepath, audio_data, self.sr)
            return filepath
        except Exception as e:
            print(f"Error saving audio clip: {e}")
            return None


class ScreamDetector:
    def __init__(self, model_path=None, threshold=None):
        """
        Initialize the scream detector
        
        Args:
            model_path (str, optional): Path to the model file
            threshold (float, optional): Detection threshold
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.threshold = threshold or Config.THRESHOLD
        self.model = None
        self.processor = AudioProcessor()
        
        # Load model if it exists
        self.load_model()
    
    def load_model(self, model_path=None):
        """
        Load the detection model
        
        Args:
            model_path (str, optional): Path to the model file
        """
        load_path = model_path or self.model_path
        
        if os.path.exists(load_path):
            try:
                # Use custom_objects to handle any custom layers that might be in the model
                self.model = load_model(load_path, compile=False)
                # Recompile the model to ensure compatibility
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                print(f"Detection model loaded from {load_path}")
                
                # Update config if custom path
                if model_path:
                    Config.MODEL_PATH = model_path
                    Config.save()
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Warning: Model file not found at {load_path}")
            return False
    
    def detect(self, audio_data=None, filepath=None):
        """
        Detect screams in audio data or file
        
        Args:
            audio_data (numpy.ndarray, optional): Raw audio data
            filepath (str, optional): Path to audio file
            
        Returns:
            tuple: (is_screaming, confidence, mel_spectrogram)
            
        Raises:
            ValueError: If neither audio_data nor filepath is provided
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        if audio_data is None and filepath is None:
            raise ValueError("Either audio_data or filepath must be provided")
            
        # Load audio if filepath is provided
        if audio_data is None:
            audio_data = self.processor.load_audio(filepath)
            
        # Preprocess audio
        if audio_data is not None:
            audio_data = self.processor.preprocess_audio(audio_data)
            
            # Extract features
            mel_spec_model, mel_spec_db = self.processor.extract_features(audio_data)
            
            try:
                # Make prediction
                prediction = self.model.predict(mel_spec_model, verbose=0)
                
                # Handle different model output formats
                if prediction.shape[1] == 2:
                    # Model outputs [not_scream_prob, scream_prob]
                    scream_prob = prediction[0][1]
                else:
                    # Model outputs single value (scream probability)
                    scream_prob = prediction[0][0]
                
                is_screaming = scream_prob > self.threshold
                
                return is_screaming, scream_prob, mel_spec_db
            except Exception as e:
                print(f"Prediction error: {e}")
                # Try reshaping the input if the model expects a different shape
                try:
                    # Try alternative shapes that models commonly expect
                    if len(mel_spec_model.shape) == 4:  # (batch, height, width, channels)
                        # Try without explicit channel dimension
                        reshaped = mel_spec_model.reshape(1, mel_spec_model.shape[1], mel_spec_model.shape[2])
                        prediction = self.model.predict(reshaped, verbose=0)
                    else:
                        # Try adding channel dimension
                        reshaped = mel_spec_model.reshape(mel_spec_model.shape + (1,))
                        prediction = self.model.predict(reshaped, verbose=0)
                    
                    # Process prediction as before
                    if prediction.shape[1] == 2:
                        scream_prob = prediction[0][1]
                    else:
                        scream_prob = prediction[0][0]
                    
                    is_screaming = scream_prob > self.threshold
                    return is_screaming, scream_prob, mel_spec_db
                except Exception as e2:
                    print(f"Alternative prediction attempt failed: {e2}")
                    # Return a safe default
                    return False, 0.0, mel_spec_db
        
        return False, 0.0, None