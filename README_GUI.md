# Chess GUI

A simple chess GUI for testing chess models.

## Features

- Interactive chess board with drag-and-drop piece movement
- Move history display
- Game status display (check, checkmate, etc.)
- Ability to undo moves
- Integration with chess AI models
- Support for loading custom models

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the GUI:

```bash
python chess_gui.py
```

## How to Use

### Playing Chess

- Click on a piece to select it
- Click on a destination square to move the selected piece
- The move will be made if it's legal
- The game status and move history will be updated automatically

### Using AI Models

- Click the "AI Move" button to make the AI play a move
- If no model is loaded, a random legal move will be made
- Click "Load Model" to load a custom model file (.h5, .pt, or .pth)

### Other Controls

- "New Game" button: Start a new game
- "Undo Move" button: Undo the last move

## Integrating Your Own Models

To integrate your own chess models with the GUI, you need to modify the `model_integration.py` file:

1. Create a wrapper class for your model that implements the `get_move(board)` method
2. Update the `ModelIntegration.load_model()` method to load your model based on the file type

Example wrapper classes for different model types are provided as commented code in the `model_integration.py` file.

## Requirements

- Python 3.6+
- chess
- Pillow
- cairosvg
- tkinter (usually comes with Python)

## Troubleshooting

If you encounter any issues:

- Make sure all dependencies are installed
- Check that your model file is compatible with the integration code
- If the GUI doesn't display properly, try resizing the window
