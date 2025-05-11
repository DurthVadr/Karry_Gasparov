# Karry_Gasparov: Chess AI with Deep Reinforcement Learning

A chess AI implementation using deep reinforcement learning techniques, with a GUI for playing against the trained models.

## Features

- **Chess GUI**: Interactive chess board with drag-and-drop piece movement
- **Deep Reinforcement Learning**: Neural network trained on high-quality chess games
- **Stockfish Integration**: Enhanced training using Stockfish evaluation
- **Prioritized Experience Replay**: Improved learning from important experiences
- **Curriculum Learning**: Gradually increasing opponent strength during training

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

4. For training on Google Colab, use the `colab_training_stockfish.ipynb` notebook

For detailed instructions on continuing training from saved models, see [MODEL_LOADING_GUIDE.md](MODEL_LOADING_GUIDE.md).

## Project Structure

- `chess_gui.py`: Interactive chess GUI
- `drl_agent.py`: Deep reinforcement learning agent implementation
- `model_integration.py`: Interface between GUI and trained models
- `train_model_stockfish.py`: Enhanced training script with Stockfish evaluation
- `data/`: Directory containing PGN files for training
- `models/`: Directory containing trained models

## References

- [Chess Engine with Reinforcement Learning](https://www.kaggle.com/code/mandmdatascience/chess-engine-2-reinforcement-learning)
- [Understanding AlphaZero](https://www.chess.com/blog/the_real_greco/understanding-alphazero-a-basic-chess-neural-network)
- [Neural Networks for Chess](https://stackoverflow.com/questions/753954/how-to-program-a-neural-network-for-chess)
- [Train Your Own Chess AI](https://medium.com/data-science/train-your-own-chess-ai-66b9ca8d71e4)

## Data Sources

High-quality chess games for training are available from:
- [Lichess Elite Database](https://database.nikonoel.fr/)
