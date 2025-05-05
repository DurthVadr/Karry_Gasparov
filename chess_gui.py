import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import chess
import chess.svg
from PIL import Image, ImageTk
import io
import cairosvg
import random
import os
import sys

# Import the model integration
from model_integration import ModelIntegration

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess GUI")
        self.root.geometry("800x600")

        # Create the chess board
        self.board = chess.Board()

        # Selected square for move input
        self.selected_square = None

        # Create the main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the board frame
        self.board_frame = ttk.Frame(self.main_frame)
        self.board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create the control frame
        self.control_frame = ttk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Create the board canvas
        self.canvas = tk.Canvas(self.board_frame, width=400, height=400, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.resize_board)
        self.canvas.bind("<Button-1>", self.on_square_click)

        # Create the status label
        self.status_var = tk.StringVar()
        self.status_var.set("White to move")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var, font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Create the move history
        self.history_frame = ttk.LabelFrame(self.control_frame, text="Move History")
        self.history_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.history_text = tk.Text(self.history_frame, width=20, height=10, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # Create the buttons
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.pack(fill=tk.X, pady=10)

        self.new_game_button = ttk.Button(self.button_frame, text="New Game", command=self.new_game)
        self.new_game_button.pack(fill=tk.X, pady=2)

        self.undo_button = ttk.Button(self.button_frame, text="Undo Move", command=self.undo_move)
        self.undo_button.pack(fill=tk.X, pady=2)

        self.ai_move_button = ttk.Button(self.button_frame, text="AI Move", command=self.ai_move)
        self.ai_move_button.pack(fill=tk.X, pady=2)

        self.load_model_button = ttk.Button(self.button_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(fill=tk.X, pady=2)

        # Initialize the board display
        self.update_board()

        # Initialize the model integration
        self.model_integration = ModelIntegration()
        self.model = None

    def resize_board(self, event=None):
        """Resize the board when the window is resized"""
        self.update_board()

    def update_board(self):
        """Update the board display"""
        # Get the SVG representation of the board
        size = min(self.canvas.winfo_width(), self.canvas.winfo_height())
        if size < 100:  # Set a minimum size to avoid errors
            size = 400

        lastmove = None
        if self.board.move_stack:
            lastmove = self.board.peek()

        svg_data = chess.svg.board(
            board=self.board,
            size=size,
            lastmove=lastmove,
            check=self.board.king(self.board.turn) if self.board.is_check() else None
        )

        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))

        # Create a PIL Image from the PNG data
        img = Image.open(io.BytesIO(png_data))

        # Convert the PIL Image to a Tkinter PhotoImage
        self.tk_img = ImageTk.PhotoImage(img)

        # Update the canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # Update status
        self.update_status()

    def update_status(self):
        """Update the status label"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_var.set(f"Checkmate! {winner} wins")
        elif self.board.is_stalemate():
            self.status_var.set("Stalemate! Draw")
        elif self.board.is_insufficient_material():
            self.status_var.set("Insufficient material! Draw")
        elif self.board.is_check():
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_var.set(f"{turn} is in check")
        else:
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_var.set(f"{turn} to move")

    def on_square_click(self, event):
        """Handle clicks on the board"""
        # Calculate the square that was clicked
        size = min(self.canvas.winfo_width(), self.canvas.winfo_height())
        square_size = size / 8

        file_idx = int(event.x / square_size)
        rank_idx = 7 - int(event.y / square_size)

        # Ensure the indices are within bounds
        if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
            square = chess.square(file_idx, rank_idx)

            # If no square is selected yet, select this one if it has a piece of the current player
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.update_board()  # Highlight the selected square
            else:
                # Try to make a move from the selected square to this one
                move = chess.Move(self.selected_square, square)

                # Check if promotion is needed
                if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                    ((self.board.turn == chess.WHITE and rank_idx == 7) or
                     (self.board.turn == chess.BLACK and rank_idx == 0))):
                    # Create a promotion move
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)

                # Check if the move is legal
                if move in self.board.legal_moves:
                    # Make the move
                    self.make_move(move)
                else:
                    # Check if it's a different piece of the same color
                    piece = self.board.piece_at(square)
                    if piece and piece.color == self.board.turn:
                        self.selected_square = square
                    else:
                        self.selected_square = None
                    self.update_board()

    def make_move(self, move):
        """Make a move on the board"""
        # Get the algebraic notation of the move
        san = self.board.san(move)

        # Make the move
        self.board.push(move)

        # Add the move to the history
        ply = len(self.board.move_stack)
        move_number = (ply + 1) // 2
        if ply % 2 == 1:  # White's move
            self.history_text.insert(tk.END, f"{move_number}. {san} ")
        else:  # Black's move
            self.history_text.insert(tk.END, f"{san}\n")

        self.history_text.see(tk.END)

        # Reset the selected square
        self.selected_square = None

        # Update the board display
        self.update_board()

    def new_game(self):
        """Start a new game"""
        self.board = chess.Board()
        self.selected_square = None
        self.history_text.delete(1.0, tk.END)
        self.update_board()

    def undo_move(self):
        """Undo the last move"""
        if self.board.move_stack:
            self.board.pop()

            # Update the history text
            text = self.history_text.get(1.0, tk.END)
            lines = text.split('\n')
            if lines[-1].strip() == '':  # Last line is empty
                if lines[-2].strip() == '':  # Second last line is also empty
                    new_text = '\n'.join(lines[:-2])
                else:
                    last_line = lines[-2]
                    if ' ' in last_line:  # Black's move exists
                        new_text = '\n'.join(lines[:-2]) + '\n' + last_line.split(' ')[0] + ' '
                    else:  # Only white's move exists
                        new_text = '\n'.join(lines[:-3])
            else:
                last_line = lines[-1]
                if ' ' in last_line:  # Black's move exists
                    new_text = '\n'.join(lines[:-1]) + '\n' + last_line.split(' ')[0] + ' '
                else:  # Only white's move exists
                    new_text = '\n'.join(lines[:-2])

            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, new_text)

            self.selected_square = None
            self.update_board()

    def ai_move(self):
        """Make a move with the AI"""
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", "The game is already over!")
            return

        # If a model is loaded, use it to make a move
        if self.model:
            try:

                move = self.get_model_move()
            except Exception as e:
                messagebox.showerror("Error", f"Error getting move from model: {str(e)}")
                move = random.choice(list(self.board.legal_moves))
        else:
            # If no model is loaded, just make a random move
            move = random.choice(list(self.board.legal_moves))

        self.make_move(move)

    def get_model_move(self):
        """Get a move from the loaded model"""
        if self.model:
            try:
                return self.model_integration.get_move(self.board)
            except Exception as e:
                print(f"Error getting move from model: {str(e)}")
                # Fall back to random move if there's an error
                return random.choice(list(self.board.legal_moves))
        else:
            # Prompt user to load a model if none is loaded
            if messagebox.askyesno("No Model Loaded", "No chess model is currently loaded. Would you like to load one now?"):
                self.load_model()
                if self.model:
                    return self.model_integration.get_move(self.board)
            # Fall back to random move
            return random.choice(list(self.board.legal_moves))

    def load_model(self):
        """Load a chess model"""
        # Open a file dialog to select the model file
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.h5 *.pt *.pth"), ("All Files", "*.*")]
        )

        if model_path:
            try:
                # Ask user for repetition penalty value
                repetition_penalty_dialog = tk.Toplevel(self.root)
                repetition_penalty_dialog.title("Repetition Penalty")
                repetition_penalty_dialog.geometry("300x150")
                repetition_penalty_dialog.transient(self.root)
                repetition_penalty_dialog.grab_set()

                # Add explanation label
                explanation = ttk.Label(
                    repetition_penalty_dialog,
                    text="Set repetition penalty (0.1-0.9):\nLower values = stronger penalty\nRecommended: 0.8",
                    wraplength=280
                )
                explanation.pack(pady=10)

                # Add slider for repetition penalty
                repetition_penalty_var = tk.DoubleVar(value=0.8)
                repetition_slider = ttk.Scale(
                    repetition_penalty_dialog,
                    from_=0.1,
                    to=0.9,
                    orient=tk.HORIZONTAL,
                    variable=repetition_penalty_var,
                    length=200
                )
                repetition_slider.pack(pady=5)

                # Add value label
                value_label = ttk.Label(repetition_penalty_dialog, text="0.8")
                value_label.pack(pady=5)

                # Update value label when slider changes
                def update_value_label(event):
                    value_label.config(text=f"{repetition_penalty_var.get():.1f}")

                repetition_slider.bind("<Motion>", update_value_label)

                # Add OK button
                ok_button = ttk.Button(
                    repetition_penalty_dialog,
                    text="OK",
                    command=repetition_penalty_dialog.destroy
                )
                ok_button.pack(pady=10)

                # Wait for dialog to close
                self.root.wait_window(repetition_penalty_dialog)

                # Get repetition penalty value
                repetition_penalty = repetition_penalty_var.get()

                # Load the model using the model integration with repetition penalty
                self.model = self.model_integration.load_model(
                    model_path,
                    repetition_penalty=repetition_penalty
                )
                messagebox.showinfo(
                    "Model Loaded",
                    f"Model loaded from {model_path}\nRepetition penalty: {repetition_penalty:.1f}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")

def main():
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
