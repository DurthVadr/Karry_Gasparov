# Karry_Gasparov Usage Guide

This guide provides detailed instructions for using the Karry_Gasparov chess AI system.

## Playing Against the AI

### Starting the GUI

1. Make sure you have installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the chess GUI:
   ```bash
   python chess_gui.py
   ```

3. The GUI will open with a new chess game. You will play as White by default.

### Loading a Model

1. Click the "Load Model" button in the GUI.
2. Navigate to the `models` directory and select one of the trained models:
   - `best_model.pt`: The best performing model during training
   - `model_pgn_final.pt`: Model trained on PGN data
   - `model_improved_final.pt`: Model with enhanced training (if available)

3. After loading the model, you can play against it by making your move and then clicking the "AI Move" button.

### GUI Controls

- **New Game**: Starts a new chess game
- **Undo Move**: Takes back the last move
- **AI Move**: Makes the AI play a move
- **Load Model**: Loads a chess model from a file

## Training Your Own Model

### Basic Training

To train a model using the enhanced training script with Stockfish evaluation:

```bash
python train_model_stockfish.py
```

This will:
1. Train on PGN files in the `data` directory
2. Continue with self-play training
3. Save models at regular intervals in the `models` directory

### Training Parameters

You can modify the following parameters in `train_model_stockfish.py`:

- `num_games`: Number of PGN games to process (default: 100)
- `num_episodes`: Number of self-play episodes (default: 500)
- `stockfish_path`: Path to the Stockfish executable

### Training on Google Colab

For training on Google Colab:

1. Upload the `colab_training_stockfish.ipynb` notebook to Google Colab
2. Upload the necessary Python files (`drl_agent.py`)
3. Follow the instructions in the notebook
4. Download the trained models after training

## Evaluating Models

To evaluate a trained model:

```bash
python evaluate_model.py --model_path models/best_model.pt
```

This will play a series of games against a random player and report the win rate.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Make sure the model file exists
   - Check that you're using the correct file format (.pt or .pth)

2. **Stockfish Not Found**:
   - Install Stockfish using your package manager
   - Update the `stockfish_path` in `train_model_stockfish.py`

3. **GUI Display Issues**:
   - Make sure you have installed all GUI dependencies
   - Try resizing the window if the board doesn't display properly

### Getting Help

If you encounter any issues, please check:
1. The documentation in the code files
2. The README.md file for general information
3. The comments in the training scripts for specific parameters
