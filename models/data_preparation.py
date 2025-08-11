"""
Data preparation module for the Scream Detection System
"""

import os
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
from ..config import Config

class ScreamingDatasetPreparation:
    def __init__(self, data_path=None, sr=None, duration=None):
        self.data_path = data_path or Config.DATA_PATH
        self.sr = sr or Config.SAMPLE_RATE
        self.duration = duration or Config.AUDIO_DURATION
        
    def load_and_preprocess_audio(self, filepath):
        """
        Load and preprocess a single audio file
        
        Args:
            filepath (str): Path to the audio file
            
        Returns:
            tuple: (mel_spectrogram, raw_audio) or (None, None) if error
        """
        try:
            y, sr = librosa.load(filepath, sr=self.sr, duration=self.duration)
            
            # Pad or trim audio to fixed length
            if len(y) < self.sr * self.duration:
                y = np.pad(y, (0, self.sr * self.duration - len(y)), 'constant')
            else:
                y = y[:self.sr * self.duration]
                
            # Extract Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db, y
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None, None
    
    def prepare_dataset(self, callback=None):
        """
        Prepare dataset from audio files in the data directory
        
        Args:
            callback (function, optional): Callback for progress updates
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        features = []
        labels = []
        
        # Process 'scream' class
        scream_path = os.path.join(self.data_path, 'scream')
        if os.path.exists(scream_path):
            scream_files = [f for f in os.listdir(scream_path) if f.endswith(('.wav', '.mp3'))]
            total_files = len(scream_files)
            
            for i, file in enumerate(scream_files):
                filepath = os.path.join(scream_path, file)
                feature, _ = self.load_and_preprocess_audio(filepath)
                if feature is not None:
                    features.append(feature)
                    labels.append(1)  # 1 for scream
                
                # Update progress
                if callback:
                    progress = (i + 1) / total_files * 50  # 0-50% for scream files
                    callback(progress, f"Processing scream files: {i+1}/{total_files}")
        
        # Process 'not_scream' class
        not_scream_path = os.path.join(self.data_path, 'not_scream')
        if os.path.exists(not_scream_path):
            not_scream_files = [f for f in os.listdir(not_scream_path) if f.endswith(('.wav', '.mp3'))]
            total_files = len(not_scream_files)
            
            for i, file in enumerate(not_scream_files):
                filepath = os.path.join(not_scream_path, file)
                feature, _ = self.load_and_preprocess_audio(filepath)
                if feature is not None:
                    features.append(feature)
                    labels.append(0)  # 0 for not scream
                
                # Update progress
                if callback:
                    progress = 50 + (i + 1) / total_files * 50  # 50-100% for not_scream files
                    callback(progress, f"Processing non-scream files: {i+1}/{total_files}")
        
        if not features:
            raise ValueError("No valid audio files found in the data directory")
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Reshape for CNN input (samples, height, width, channels)
        features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
        
        # One-hot encode labels
        labels = to_categorical(labels, 2)
        
        return features, labels