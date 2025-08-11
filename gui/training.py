"""
Training tab for the Scream Detection System GUI
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf

from ..config import Config
from ..models.data_preparation import ScreamingDatasetPreparation
from ..models.detection_model import ScreamingDetectionModel

class TrainingTab(ttk.Frame):
    def __init__(self, parent):
        """
        Initialize training tab
        
        Args:
            parent (ttk.Notebook): Parent notebook widget
        """
        super().__init__(parent, padding=10)
        
        # Initialize training components
        self.data_prep = None
        self.model = None
        self.training_thread = None
        
        # Training params
        self.epochs = 30
        self.batch_size = 32
        self.validation_split = 0.2
        
        # Create UI components
        self.create_widgets()
    
    def create_widgets(self):
        """Create tab widgets"""
        # Data section
        data_frame = ttk.LabelFrame(self, text="Training Data")
        data_frame.pack(fill=tk.X, pady=10)
        
        # Data path
        data_path_frame = ttk.Frame(data_frame)
        data_path_frame.pack(fill=tk.X, padx=10, pady=5)
        
        data_path_label = ttk.Label(data_path_frame, text="Data Directory:")
        data_path_label.pack(side=tk.LEFT, padx=5)
        
        self.data_path_var = tk.StringVar(value=Config.DATA_PATH)
        data_path_entry = ttk.Entry(data_path_frame, textvariable=self.data_path_var, width=50)
        data_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_data_btn = ttk.Button(data_path_frame, text="Browse", command=self.browse_data_path)
        browse_data_btn.pack(side=tk.LEFT, padx=5)
        
        # Data structure explanation
        data_info_frame = ttk.Frame(data_frame)
        data_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        data_info_label = ttk.Label(data_info_frame, 
                                   text="Expected data structure:\n"
                                        "  • data/\n"
                                        "    ├── scream/      (WAV/MP3 files of screams)\n"
                                        "    └── not_scream/  (WAV/MP3 files of other sounds)",
                                   justify=tk.LEFT)
        data_info_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Training parameters
        param_frame = ttk.LabelFrame(self, text="Training Parameters")
        param_frame.pack(fill=tk.X, pady=10)
        
        # Epochs
        epochs_frame = ttk.Frame(param_frame)
        epochs_frame.pack(fill=tk.X, padx=10, pady=5)
        
        epochs_label = ttk.Label(epochs_frame, text="Epochs:")
        epochs_label.pack(side=tk.LEFT, padx=5)
        
        self.epochs_var = tk.IntVar(value=30)
        epochs_spinbox = ttk.Spinbox(epochs_frame, from_=5, to=100, 
                                    increment=5, textvariable=self.epochs_var, 
                                    width=10)
        epochs_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Batch size
        batch_frame = ttk.Frame(param_frame)
        batch_frame.pack(fill=tk.X, padx=10, pady=5)
        
        batch_label = ttk.Label(batch_frame, text="Batch Size:")
        batch_label.pack(side=tk.LEFT, padx=5)
        
        self.batch_size_var = tk.IntVar(value=32)
        batch_sizes = [8, 16, 32, 64, 128]
        batch_combo = ttk.Combobox(batch_frame, textvariable=self.batch_size_var, 
                                   values=batch_sizes, state="readonly", width=10)
        batch_combo.pack(side=tk.LEFT, padx=5)
        
        # Validation split
        val_frame = ttk.Frame(param_frame)
        val_frame.pack(fill=tk.X, padx=10, pady=5)
        
        val_label = ttk.Label(val_frame, text="Validation Split:")
        val_label.pack(side=tk.LEFT, padx=5)
        
        self.val_split_var = tk.DoubleVar(value=0.2)
        val_split_spinbox = ttk.Spinbox(val_frame, from_=0.1, to=0.5, 
                                       increment=0.05, textvariable=self.val_split_var, 
                                       width=10)
        val_split_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.prepare_btn = ttk.Button(button_frame, text="Prepare Dataset", 
                                     command=self.prepare_dataset)
        self.prepare_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(button_frame, text="Train Model", 
                                   command=self.start_training,
                                   state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Model", 
                                  command=self.save_model,
                                  state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Training progress
        progress_frame = ttk.LabelFrame(self, text="Training Progress")
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, 
                                           length=400, mode='determinate', 
                                           variable=self.progress_var)
        self.progress_bar.pack(padx=10, pady=10, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, 
                               anchor=tk.W, wraplength=600)
        status_label.pack(padx=10, pady=5, fill=tk.X)
        
        # Results visualization
        viz_frame = ttk.LabelFrame(self, text="Training Results")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_figure = Figure(figsize=(6, 4), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, viz_frame)
        self.result_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def browse_data_path(self):
        """Browse for data directory"""
        data_path = filedialog.askdirectory(
            title="Select Data Directory",
            initialdir=os.path.dirname(Config.DATA_PATH)
        )
        if data_path:
            self.data_path_var.set(data_path)
            
            # Update config
            Config.DATA_PATH = data_path
            Config.save()
    
    def prepare_dataset(self):
        """Prepare the dataset for training"""
        data_path = self.data_path_var.get()
        
        # Check if data path exists
        if not os.path.exists(data_path):
            messagebox.showerror("Error", "Data directory not found")
            return
        
        # Check for required subdirectories
        scream_dir = os.path.join(data_path, "scream")
        not_scream_dir = os.path.join(data_path, "not_scream")
        
        if not os.path.exists(scream_dir) or not os.path.exists(not_scream_dir):
            should_create = messagebox.askyesno(
                "Warning", 
                "Missing required subdirectories (scream and/or not_scream).\n"
                "Would you like to create them?"
            )
            
            if should_create:
                # Create directories if confirmed
                os.makedirs(scream_dir, exist_ok=True)
                os.makedirs(not_scream_dir, exist_ok=True)
                
                messagebox.showinfo("Info", 
                                "Directories created. Please add audio files to:\n"
                                f"- {scream_dir}\n"
                                f"- {not_scream_dir}\n\n"
                                "Then click 'Prepare Dataset' again.")
            return
        
        # Check for audio files
        scream_files = [f for f in os.listdir(scream_dir) if f.endswith(('.wav', '.mp3'))]
        not_scream_files = [f for f in os.listdir(not_scream_dir) if f.endswith(('.wav', '.mp3'))]
        
        if not scream_files:
            messagebox.showerror("Error", f"No audio files found in {scream_dir}")
            return
            
        if not not_scream_files:
            messagebox.showerror("Error", f"No audio files found in {not_scream_dir}")
            return
        
        # Update UI
        self.status_var.set("Preparing dataset...")
        self.progress_var.set(0)
        self.update()
        
        # Disable buttons during processing
        self.prepare_btn.configure(state=tk.DISABLED)
        
        # Create data preparation object
        self.data_prep = ScreamingDatasetPreparation(data_path)
        
        # Start preparation in a separate thread
        threading.Thread(target=self._prepare_dataset_thread, daemon=True).start()
    
    def _prepare_dataset_thread(self):
        """Background thread for dataset preparation"""
        try:
            # Prepare dataset with progress callback
            features, labels = self.data_prep.prepare_dataset(
                callback=self._update_progress
            )
            
            # Store prepared data
            self.features = features
            self.labels = labels
            
            # Update UI on main thread
            self.after(10, self._on_dataset_prepared)
            
        except Exception as e:
            # Update UI on main thread with error
            self.after(10, lambda: self._on_dataset_error(str(e)))
    
    def _update_progress(self, progress, status_text=None):
        """Update progress bar during dataset preparation"""
        self.after(10, lambda: self._update_progress_ui(progress, status_text))
    
    def _update_progress_ui(self, progress, status_text=None):
        """Update UI with progress information"""
        self.progress_var.set(progress)
        if status_text:
            self.status_var.set(status_text)
    
    def _on_dataset_prepared(self):
        """Called when dataset preparation is complete"""
        # Update UI
        num_samples = len(self.features)
        scream_samples = np.sum(np.argmax(self.labels, axis=1))
        not_scream_samples = num_samples - scream_samples
        
        self.status_var.set(f"Dataset prepared: {num_samples} total samples "
                           f"({scream_samples} scream, {not_scream_samples} not scream)")
        self.progress_var.set(100)
        
        # Enable training
        self.prepare_btn.configure(state=tk.NORMAL)
        self.train_btn.configure(state=tk.NORMAL)
        
        # Create model
        input_shape = self.features[0].shape
        self.model = ScreamingDetectionModel(input_shape)
        
        # Show sample visualization
        self._visualize_samples()
    
    def _on_dataset_error(self, error_message):
        """Called when dataset preparation fails"""
        self.status_var.set(f"Error: {error_message}")
        self.prepare_btn.configure(state=tk.NORMAL)
        messagebox.showerror("Error", f"Failed to prepare dataset: {error_message}")
    
    def _visualize_samples(self):
        """Visualize sample spectrograms"""
        if not hasattr(self, 'features') or len(self.features) == 0:
            return
            
        # Clear figure
        self.result_figure.clear()
        
        # Find one example of each class
        scream_idx = np.where(np.argmax(self.labels, axis=1) == 1)[0][0]
        not_scream_idx = np.where(np.argmax(self.labels, axis=1) == 0)[0][0]
        
        # Create subplots
        ax1 = self.result_figure.add_subplot(121)
        ax2 = self.result_figure.add_subplot(122)
        
        # Plot spectrograms
        scream_spec = self.features[scream_idx].squeeze()
        not_scream_spec = self.features[not_scream_idx].squeeze()
        
        ax1.imshow(scream_spec, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title("Scream Sample")
        ax1.set_ylabel("Mel Frequency Bin")
        ax1.set_xlabel("Time")
        
        ax2.imshow(not_scream_spec, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title("Non-Scream Sample")
        ax2.set_ylabel("Mel Frequency Bin")
        ax2.set_xlabel("Time")
        
        self.result_figure.tight_layout()
        self.result_canvas.draw()
    
    def start_training(self):
        """Start model training"""
        if not hasattr(self, 'features') or not hasattr(self, 'labels'):
            messagebox.showwarning("Warning", "Please prepare the dataset first")
            return
            
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Info", "Training is already in progress")
            return
            
        # Get training parameters
        self.epochs = self.epochs_var.get()
        self.batch_size = self.batch_size_var.get()
        self.validation_split = self.val_split_var.get()
        
        # Update UI
        self.status_var.set("Training model...")
        self.progress_var.set(0)
        
        # Disable buttons during training
        self.prepare_btn.configure(state=tk.DISABLED)
        self.train_btn.configure(state=tk.DISABLED)
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self._train_model_thread, daemon=True)
        self.training_thread.start()
    
    def _train_model_thread(self):
        """Background thread for model training"""
        try:
            # Custom callback for training progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, parent):
                    super().__init__()
                    self.parent = parent
                    
                def on_epoch_end(self, epoch, logs=None):
                    # Calculate progress as percentage of epochs
                    progress = (epoch + 1) / self.parent.epochs * 100
                    status_text = f"Epoch {epoch+1}/{self.parent.epochs} - "
                    status_text += f"loss: {logs.get('loss'):.4f}, "
                    status_text += f"accuracy: {logs.get('accuracy'):.4f}, "
                    status_text += f"val_accuracy: {logs.get('val_accuracy'):.4f}"
                    
                    # Update UI on main thread
                    self.parent.after(10, lambda: self.parent._update_progress_ui(progress, status_text))
            
            # Create callback
            progress_cb = ProgressCallback(self)
            
            # Train model
            history, accuracy = self.model.train(
                self.features, self.labels,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[progress_cb]
            )
            
            # Store history
            self.training_history = history
            
            # Update UI on main thread
            self.after(10, self._on_training_complete)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Update UI on main thread with error
            self.after(10, lambda: self._on_training_error(str(e)))
    
    def _on_training_complete(self):
        """Called when training is complete"""
        # Update UI
        self.status_var.set("Training complete!")
        self.progress_var.set(100)
        
        # Enable buttons
        self.prepare_btn.configure(state=tk.NORMAL)
        self.train_btn.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.NORMAL)
        
        # Visualize training results
        self._visualize_training_results()
        
        # Show completion message
        messagebox.showinfo("Training Complete", 
                           "Model training completed successfully.\n"
                           "You can now save the model.")
    
    def _on_training_error(self, error_message):
        """Called when training fails"""
        self.status_var.set(f"Error: {error_message}")
        
        # Re-enable buttons
        self.prepare_btn.configure(state=tk.NORMAL)
        self.train_btn.configure(state=tk.NORMAL)
        
        messagebox.showerror("Error", f"Training failed: {error_message}")
    
    def _visualize_training_results(self):
        """Visualize training history"""
        if not hasattr(self, 'training_history'):
            return
            
        history = self.training_history
        
        # Clear figure
        self.result_figure.clear()
        
        # Create subplots
        ax1 = self.result_figure.add_subplot(121)
        ax2 = self.result_figure.add_subplot(122)
        
        # Plot training & validation accuracy
        if 'accuracy' in history.history:
            ax1.plot(history.history['accuracy'], label='Training')
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='lower right')
        ax1.set_ylim([0, 1.0])
        ax1.grid(True)
        
        # Plot training & validation loss
        if 'loss' in history.history:
            ax2.plot(history.history['loss'], label='Training')
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        self.result_figure.tight_layout()
        self.result_canvas.draw()
    
    def save_model(self):
        """Save the trained model"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showwarning("Warning", "No model to save")
            return
            
        # Ask for confirmation
        save_path = Config.MODEL_PATH
        if os.path.exists(save_path):
            if not messagebox.askyesno("Confirm", 
                                     f"This will overwrite the existing model at:\n{save_path}\n\nProceed?"):
                return
                
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model
            self.model.save_model(save_path)
            
            # Update UI
            self.status_var.set(f"Model saved to {save_path}")
            
            # Show success message
            messagebox.showinfo("Success", 
                               f"Model saved successfully to:\n{save_path}")
                               
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")