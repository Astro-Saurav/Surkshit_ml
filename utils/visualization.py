"""
Visualization utilities for the Scream Detection System
"""

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

class AudioVisualizer:
    def __init__(self, figure=None, canvas=None):
        """
        Initialize audio visualizer
        
        Args:
            figure (matplotlib.figure.Figure, optional): Figure to draw on
            canvas (FigureCanvasTkAgg, optional): Canvas for the figure
        """
        self.figure = figure
        self.canvas = canvas
    
    def plot_waveform(self, audio_data, sample_rate=16000, title="Audio Waveform", ax=None):
        """
        Plot audio waveform
        
        Args:
            audio_data (numpy.ndarray): Audio data
            sample_rate (int, optional): Sample rate
            title (str, optional): Plot title
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            
        Returns:
            matplotlib.axes.Axes: The axes that was plotted on
        """
        if ax is None:
            if self.figure is None:
                _, ax = plt.subplots(figsize=(10, 4))
            else:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
        
        # Calculate time axis in seconds
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        # Plot waveform
        ax.plot(time, audio_data)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True)
        
        # Set reasonable y-axis limits
        max_amp = max(0.1, np.max(np.abs(audio_data)))
        ax.set_ylim(-max_amp*1.1, max_amp*1.1)
        
        if self.figure is not None and self.canvas is not None:
            self.figure.tight_layout()
            self.canvas.draw()
        
        return ax
    
    def plot_melspectrogram(self, mel_spec_db, sample_rate=16000, title="Mel Spectrogram (dB)", ax=None):
        """
        Plot mel spectrogram
        
        Args:
            mel_spec_db (numpy.ndarray): Mel spectrogram in dB
            sample_rate (int, optional): Sample rate
            title (str, optional): Plot title
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            
        Returns:
            tuple: (matplotlib.axes.Axes, matplotlib.colorbar.Colorbar)
        """
        if ax is None:
            if self.figure is None:
                _, ax = plt.subplots(figsize=(10, 4))
            else:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
        
        # Plot mel spectrogram
        img = librosa.display.specshow(
            mel_spec_db, 
            x_axis='time', 
            y_axis='mel', 
            sr=sample_rate,
            fmax=8000,
            ax=ax
        )
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title)
        
        if self.figure is not None and self.canvas is not None:
            self.figure.tight_layout()
            self.canvas.draw()
        
        return ax, cbar
    
    def update_progress(self, progress_var, status_var, progress, status_text=None):
        """
        Update progress bar and status text
        
        Args:
            progress_var (tkinter.DoubleVar): Progress variable
            status_var (tkinter.StringVar): Status text variable
            progress (float): Progress value (0-100)
            status_text (str, optional): Status text
        """
        progress_var.set(progress)
        if status_text:
            status_var.set(status_text)
    
    def update_confidence_display(self, confidence_var, confidence_text, confidence):
        """
        Update confidence display
        
        Args:
            confidence_var (tkinter.DoubleVar): Confidence variable for progress bar
            confidence_text (tkinter.Label): Label to show confidence percentage
            confidence (float): Confidence value (0-1)
        """
        confidence_var.set(confidence * 100)
        confidence_text.config(text=f"{confidence*100:.1f}%")