# Karry_Gasparov: Chess AI with Deep Reinforcement Learning

This repository contains a chess AI implementation using deep reinforcement learning techniques. The system includes a chess GUI, a deep reinforcement learning agent, and training methods using both PGN data and self-play.

## Refactored Code Structure

The codebase has been refactored for better organization, readability, and maintainability:

```
Karry_Gasparov/
├── chess_gui.py           # Interactive chess GUI
├── drl_agent.py           # Deep reinforcement learning agent
├── model_integration.py   # Interface between GUI and models
├── memory.py              # Replay memory implementations
├── reward.py              # Reward calculation functions
├── evaluation.py          # Model evaluation against Stockfish
├── visualization.py       # Training progress visualization
├── training.py            # Training methods implementation
├── main.py                # Main script with command-line interface
├── train_model_stockfish_refactored.py  # Refactored training script
├── data/                  # Directory for PGN files
├── models/                # Directory for saved models
└── COMPREHENSIVE_GUIDE.md # Detailed documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Karry_Gasparov.git
   cd Karry_Gasparov
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Stockfish (optional, for enhanced training):
   - macOS: `brew install stockfish`
   - Ubuntu/Debian: `apt-get install stockfish`
   - Windows: Download from [Stockfish website](https://stockfishchess.org/download/)

## Usage

### Training a Model

Use the main.py script to train a model with various options:

```bash
# Train from PGN data
python main.py --data_dir data --num_games 10000 --batch_size 64 --save_interval 1000

# Train with self-play after PGN training
python main.py --data_dir data --num_games 5000 --self_play --self_play_episodes 2000 --stockfish_path /usr/local/bin/stockfish

# Evaluate a trained model against Stockfish
python main.py --evaluate --eval_games 5 --min_level 1 --max_level 7 --stockfish_path /usr/local/bin/stockfish
```

### Command-line Options

The main.py script provides the following options:

| Option | Description | Default |
|--------|-------------|---------|
| --data_dir | Directory containing PGN files | data |
| --num_games | Maximum number of games to process | 200000 |
| --batch_size | Batch size for training | 64 |
| --save_interval | How often to save the model (in games) | 1000 |
| --self_play | Run self-play training after PGN training | False |
| --self_play_episodes | Number of self-play episodes | 5000 |
| --stockfish_path | Path to Stockfish executable | None |
| --stockfish_opponent | Use Stockfish as opponent in self-play | True |
| --target_level | Target Stockfish level to achieve | 7 |
| --eval_interval | How often to evaluate against target level | 500 |
| --evaluate | Evaluate model against Stockfish levels | False |
| --eval_games | Number of games to play against each level | 5 |
| --eval_model | Path to specific model to evaluate | None |
| --min_level | Minimum Stockfish level to test against | 1 |
| --max_level | Maximum Stockfish level to test against | 10 |

### Playing Against the Model

Use the chess_gui.py script to play against a trained model:

```bash
# Start the GUI with a specific model
python chess_gui.py models/best_model.pt

# Start the GUI without a model (you can load one later)
python chess_gui.py
```

## Module Descriptions

### memory.py

Contains implementations of replay memory for storing and sampling experiences:
- `ReplayMemory`: Standard uniform random sampling
- `PrioritizedReplayMemory`: Sampling based on TD error priority

### reward.py

Provides functions for calculating rewards based on chess positions:
- `RewardCalculator`: Calculates rewards using Stockfish evaluation or material advantage

### evaluation.py

Contains functions for evaluating chess models against Stockfish:
- `ModelEvaluator`: Evaluates models against different Stockfish levels

### visualization.py

Provides functions for visualizing training progress and model performance:
- `plot_training_progress`: Plots training metrics
- `plot_evaluation_results`: Plots evaluation results against Stockfish

### training.py

Contains classes for training chess models:
- `PGNTrainer`: Training from PGN files
- `SelfPlayTrainer`: Training through self-play

### main.py

Main script with command-line interface for training and evaluating models.

## Comprehensive Documentation

For detailed documentation on all components and options, see the [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md) file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
