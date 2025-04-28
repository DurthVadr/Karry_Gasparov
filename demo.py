"""
Demo script to show how to use the chess GUI with a model.
"""

import tkinter as tk
from chess_gui import ChessGUI
from model_integration import RandomModel

def main():
    # Create the root window
    root = tk.Tk()
    root.title("Chess GUI Demo")
    
    # Create the GUI
    app = ChessGUI(root)
    
    # Load a random model (for demonstration)
    app.model = RandomModel()
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
