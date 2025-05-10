# Chess Model Evaluation Guide

This guide explains how to evaluate chess models using the provided evaluation tools.

## Stockfish Path Configuration

You need to specify the correct Stockfish path for your operating system:

- **macOS**: `/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish`
- **Windows**: `C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe`

Make sure to use the correct path for your system when running the evaluation commands.

## Automatic Evaluation During Training

The training system now automatically evaluates checkpoint models during training:

1. **PGN Training**: Each time a checkpoint model is saved (controlled by `save_interval`), the model is evaluated against Stockfish levels 1, 3, and 5.

2. **Self-Play Training**: Each time a checkpoint model is saved, the model is evaluated against Stockfish levels 1, 3, 5, and the target level.

This automatic evaluation provides immediate feedback on model performance without waiting for the full training to complete.

## Comprehensive Evaluation Script

For more detailed evaluation after training, use the `evaluate_chess_model.py` script, which provides three evaluation modes:

### 1. Stockfish Evaluation Mode

Evaluates a model against Stockfish at different skill levels:

```bash
# macOS
python evaluate_chess_model.py --model_path models/your_model.pt --mode stockfish --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish --num_games 10 --min_level 1 --max_level 10

# Windows
python evaluate_chess_model.py --model_path models/your_model.pt --mode stockfish --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe" --num_games 10 --min_level 1 --max_level 10
```

Options:
- `--model_path`: Path to the model file to evaluate (required)
- `--stockfish_path`: Path to Stockfish executable (required for stockfish mode)
- `--num_games`: Number of games to play against each Stockfish level (default: 10)
- `--min_level`: Minimum Stockfish level to test against (default: 1)
- `--max_level`: Maximum Stockfish level to test against (default: 10)
- `--output_dir`: Directory to save evaluation results (default: "evaluation_results")
- `--no_plots`: Don't save evaluation plots

### 2. Position Evaluation Mode

Evaluates a model on random chess positions:

```bash
python evaluate_chess_model.py --model_path models/your_model.pt --mode positions --num_positions 5 --num_moves 5
```

Options:
- `--model_path`: Path to the model file to evaluate (required)
- `--num_positions`: Number of random positions to evaluate (default: 5)
- `--num_moves`: Number of moves to play from each position (default: 5)
- `--output_dir`: Directory to save evaluation results (default: "evaluation_results")

### 3. Model Comparison Mode

Compares multiple models on the same positions:

```bash
python evaluate_chess_model.py --model_path models/your_model.pt --mode compare --compare_with models/model1.pt models/model2.pt --num_positions 3 --num_moves 5
```

Options:
- `--model_path`: Path to the primary model file to evaluate (required)
- `--compare_with`: Paths to other model files to compare with (required for compare mode)
- `--num_positions`: Number of random positions to evaluate (default: 3)
- `--num_moves`: Number of moves to play from each position (default: 5)
- `--output_dir`: Directory to save evaluation results (default: "evaluation_results")

## Evaluation Results

All evaluation results are saved in the specified output directory (default: "evaluation_results"):

1. **Stockfish Evaluation**:
   - Text file with detailed results for each Stockfish level
   - Plot showing win rate and score against different Stockfish levels

2. **Position Evaluation**:
   - Text file showing the model's moves from each position

3. **Model Comparison**:
   - Text file comparing the moves of different models from the same positions

## Example Workflow

1. **Train a model with PGN data and self-play**:
   ```bash
   # macOS
   python main.py --data_dir data --num_games 5000 --self_play --self_play_episodes 2000 --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish

   # Windows
   python main.py --data_dir data --num_games 5000 --self_play --self_play_episodes 2000 --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe"
   ```

   This command will:
   - Train on up to 5000 PGN games from the data directory
   - Then perform 2000 episodes of self-play training
   - Automatically evaluate checkpoints during training

2. **Evaluate against Stockfish**:

   You can evaluate using either the main.py script or the dedicated evaluate_chess_model.py script:

   Using main.py (simpler):
   ```bash
   # macOS
   python main.py --evaluate --eval_model models/best_model.pt --eval_games 20 --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish

   # Windows
   python main.py --evaluate --eval_model models/best_model.pt --eval_games 20 --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe"
   ```

   Using evaluate_chess_model.py (more options):
   ```bash
   # macOS
   python evaluate_chess_model.py --model_path models/best_model.pt --mode stockfish --stockfish_path /opt/homebrew/Cellar/stockfish/17.1/bin/stockfish --num_games 20

   # Windows
   python evaluate_chess_model.py --model_path models/best_model.pt --mode stockfish --stockfish_path "C:\\Users\\Can\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe" --num_games 20
   ```

3. **Compare with previous models**:
   ```bash
   python evaluate_chess_model.py --model_path models/best_model.pt --mode compare --compare_with models/model_pgn_final.pt models/model_self_play_final.pt
   ```

4. **Analyze on specific positions**:
   ```bash
   python evaluate_chess_model.py --model_path models/best_model.pt --mode positions --num_positions 10 --num_moves 10
   ```

## Interpreting Results

### Stockfish Evaluation

- **Win Rate**: Percentage of games won against Stockfish
- **Score**: (wins + 0.5*draws) / total games
- **Model Strength**: A model is considered competitive with a Stockfish level if it achieves a score of 45% or higher

### Position Evaluation

- Look for sensible moves that follow chess principles
- Check if the model avoids obvious blunders
- Observe how the model handles different types of positions

### Model Comparison

- Compare how different models handle the same positions
- Identify which model makes more sensible moves
- Observe differences in playing style between models
