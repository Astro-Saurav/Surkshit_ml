"""
Model definition and training for the Scream Detection System
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from ..config import Config

class ScreamingDetectionModel:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.model = None
        if input_shape:
            self.model = self.build_model()
        
    def build_model(self):
        """
        Build and compile the CNN model
        
        Returns:
            tensorflow.keras.models.Sequential: Compiled model
        """
        model = Sequential([
            # Convolutional layers
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, features, labels, epochs=30, batch_size=32, validation_split=0.2, callbacks=None):
        """
        Train the model on the provided dataset
        
        Args:
            features (numpy.ndarray): Input features
            labels (numpy.ndarray): Target labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            callbacks (list): List of Keras callbacks
            
        Returns:
            tuple: (history, accuracy) of training history and final test accuracy
        """
        if self.model is None:
            if self.input_shape is None and len(features) > 0:
                self.input_shape = features[0].shape
            self.model = self.build_model()
            
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42
        )
        
        # Create callbacks
        all_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        if callbacks:
            all_callbacks.extend(callbacks)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=all_callbacks
        )
        
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_val, y_val)
        
        return history, test_accuracy
    
    def evaluate(self, X_test, y_test, save_path=None):
        """
        Evaluate the model and generate reports
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            save_path (str, optional): Path to save evaluation plots
            
        Returns:
            float: Test accuracy
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model or load_model first.")
            
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Print classification report
        print(classification_report(y_true_classes, y_pred_classes, target_names=['Not Scream', 'Scream']))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Scream', 'Scream'],
                   yticklabels=['Not Scream', 'Scream'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
        return accuracy
    
    def save_model(self, filepath=None):
        """
        Save the model to disk
        
        Args:
            filepath (str, optional): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        save_path = filepath or Config.MODEL_PATH
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
        # Update config
        Config.MODEL_PATH = save_path
        Config.save()
    
    def load_model(self, filepath=None):
        """
        Load a model from disk
        
        Args:
            filepath (str, optional): Path to the model file
        """
        load_path = filepath or Config.MODEL_PATH
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        self.model = load_model(load_path)
        print(f"Model loaded from {load_path}")
        
        # Update config
        if filepath:
            Config.MODEL_PATH = filepath
            Config.save()