# Karry_Gasparov: Chess AI with Deep Reinforcement Learning

A chess AI implementation using deep reinforcement learning techniques, with a GUI for playing against the trained models.

## Features

- **Chess GUI**: Interactive chess board with drag-and-drop piece movement
- **Deep Reinforcement Learning**: Neural network trained on high-quality chess games
- **Stockfish Integration**: Enhanced training using Stockfish evaluation
- **Prioritized Experience Replay**: Improved learning from important experiences
- **Curriculum Learning**: Gradually increasing opponent strength during training
- **Comprehensive Metrics System**: Advanced metrics for model quality evaluation beyond loss values
- **Phase-Specific Performance Analysis**: Evaluate model strength in opening, middlegame, and endgame
- **Asynchronous Evaluation**: Lightweight metrics that don't impact training performance

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DurthVadr/Karry_Gasparov.git
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

### Playing Against the AI

1. Start the chess GUI:
   ```bash
   python chess_gui.py
   ```

2. Load a trained model by clicking "Load Model" and selecting a model file (e.g., `models/best_model.pt`)

3. Play as White by making your moves, then click "AI Move" to have the model respond as Black

### Training Your Own Model

1. Train using PGN data and Stockfish evaluation:
   ```bash
   python main.py --data_dir data --stockfish_path /path/to/stockfish
   ```

2. Continue training with self-play:
   ```bash
   python main.py --self_play --stockfish_path /path/to/stockfish
   ```

3. Continue training from a saved model:
   ```bash
   python main.py --load_model models/model_pgn_checkpoint.pt --data_dir data
   ```

4. Evaluate model metrics:
   ```bash
   python main.py --load_model models/best_model.pt --metrics_only
   ```

5. For training on Google Colab, upload the necessary Python files and use the notebook in the archive directory

For detailed instructions on continuing training from saved models, see [MODEL_LOADING_GUIDE.md](MODEL_LOADING_GUIDE.md).

## Model Metrics System

The project includes a comprehensive metrics system that provides deeper insights into model quality beyond simple loss values:

### Available Metrics

- **Decision Quality Index**: Measures how often the model makes better moves than random play
- **Tactical Accuracy**: Evaluates the model's performance on tactical positions (captures, checks, etc.)
- **Phase-Specific Performance**: Separate metrics for opening, middlegame, and endgame play
- **Evaluation Correlation**: Measures how well the model's evaluations correlate with material count

### Using the Metrics System

1. **During Training**: Metrics are automatically evaluated and visualized at regular intervals
   ```bash
   # Metrics are included in normal training
   python main.py --self_play --stockfish_path /path/to/stockfish
   ```

2. **Standalone Evaluation**: Evaluate metrics on a previously trained model
   ```bash
   python main.py --load_model models/best_model.pt --metrics_only
   ```

3. **Visualizations**: Metrics visualizations are saved to the `models/metrics/` directory:
   - `metrics_history.png`: Shows how metrics change over time
   - `metrics_radar.png`: Radar chart showing strengths and weaknesses

### Benefits Over Loss Values

- **Chess-Specific Insights**: Metrics directly measure chess understanding, not just pattern recognition
- **Phase Analysis**: Identify if your model is weak in specific phases (opening, middlegame, endgame)
- **Trend Tracking**: See if your model is improving, stagnating, or declining in specific areas
- **Lightweight Evaluation**: Most metrics don't require Stockfish, making them faster to compute

## Project Structure

- `chess_gui.py`: Interactive chess GUI
- `drl_agent.py`: Deep reinforcement learning agent implementation
- `model_integration.py`: Interface between GUI and trained models
- `main.py`: Main training script with command-line interface
- `memory.py`: Replay memory implementations
- `reward.py`: Reward calculation functions
- `evaluation.py`: Model evaluation against Stockfish
- `visualization.py`: Training progress visualization
- `training.py`: Training methods implementation
- `utils.py`: Utility functions
- `metrics.py`: Advanced metrics for model quality evaluation
- `metrics_visualization.py`: Visualization tools for model metrics
- `data/`: Directory containing PGN files for training
- `models/`: Directory containing trained models
- `models/metrics/`: Directory containing metrics visualizations

## References

- [Chess Engine with Reinforcement Learning](https://www.kaggle.com/code/mandmdatascience/chess-engine-2-reinforcement-learning)
- [Understanding AlphaZero](https://www.chess.com/blog/the_real_greco/understanding-alphazero-a-basic-chess-neural-network)
- [Neural Networks for Chess](https://stackoverflow.com/questions/753954/how-to-program-a-neural-network-for-chess)
- [Train Your Own Chess AI](https://medium.com/data-science/train-your-own-chess-ai-66b9ca8d71e4)

## Data Sources

High-quality chess games for training are available from:
- [Lichess Elite Database](https://database.nikonoel.fr/)
