import tkinter as tk
from tkinter import ttk

class BaseTab(ttk.Frame):
    def __init__(self, parent):
        """
        Base initialization for all tabs
        
        Args:
            parent (ttk.Notebook): Parent notebook widget
        """
        super().__init__(parent, padding=10)
        
        # Store root window reference
        self.root = parent.winfo_toplevel()
        
        # Common initialization
        self.create_widgets()
    
    def create_widgets(self):
        """
        Create tab-specific widgets
        
        Override this method in child classes
        """
        pass