# Enhanced Chess Training System

This system implements a deep reinforcement learning approach to train a chess AI using Stockfish for evaluation and guidance. The system uses a combination of self-play, curriculum learning, and Stockfish evaluation to create a strong chess-playing model.

## System Architecture

### Core Components

1. **Neural Network (DQN)**
   - Deep Q-Network architecture
   - Input: Chess board state (8x8x12 tensor)
   - Output: Q-values for all possible moves
   - Uses residual connections and batch normalization

2. **Training System**
   - Prioritized Experience Replay
   - Adaptive learning rate scheduling
   - Curriculum learning
   - Stockfish integration for evaluation

3. **Evaluation System**
   - Stockfish-based position evaluation
   - Strength testing against different Stockfish levels
   - Comprehensive progress monitoring

## Training Process

### 1. Curriculum Learning Stages

The training progresses through four stages:

1. **Opening Stage** (20 moves)
   - Focus: Basic opening principles
   - Opponent strength: 20% of full strength
   - Stockfish depth: 8

2. **Middlegame Stage** (40 moves)
   - Focus: Tactical play and piece coordination
   - Opponent strength: 40% of full strength
   - Stockfish depth: 10

3. **Endgame Stage** (60 moves)
   - Focus: Endgame techniques and pawn structures
   - Opponent strength: 60% of full strength
   - Stockfish depth: 11

4. **Full Game Stage** (100 moves)
   - Focus: Complete game understanding
   - Opponent strength: 80% of full strength
   - Stockfish depth: 12

### 2. Training Components

#### Experience Replay
- Capacity: 100,000 positions
- Prioritized sampling based on TD errors
- Batch size: 1024
- Importance sampling weights for stability

#### Learning Rate Management
- Initial learning rate: 0.001
- Adaptive scheduling using ReduceLROnPlateau
- Patience: 10 episodes
- Minimum learning rate: 1e-6

#### Exploration Strategy
- Initial epsilon: 1.0
- Final epsilon: 0.05
- Adaptive adjustment based on performance
- Decay over 10,000 steps

### 3. Reward System

The reward function combines multiple components:

1. **Stockfish Evaluation** (70% weight)
   - Dynamic depth based on game phase
   - Scaled based on position complexity
   - Mate score handling

2. **Positional Bonuses** (30% weight)
   - Center control
   - Piece development
   - King safety
   - Pawn structure

### 4. Progress Monitoring

The system tracks multiple metrics:

1. **Training Metrics**
   - Episode rewards
   - Game lengths
   - Loss values
   - Q-value statistics

2. **Performance Metrics**
   - Win rates against Stockfish
   - Model strength level
   - Learning rate changes
   - Exploration rate

3. **Checkpointing**
   - Saves model every 100 episodes
   - Stores training statistics
   - Maintains best model

## Usage

### Training

```python
# Initialize trainer
trainer = ChessTrainerWithStockfish(
    model_dir="models",
    stockfish_path="/path/to/stockfish",
    stockfish_level=8  # Training level
)

# Start training
rewards, lengths, losses = trainer.train_self_play(num_episodes=1000)
```

### Testing Model Strength

```python
# Test model against different Stockfish levels
strength = trainer.test_model_strength(num_games=10)
print(f"Model strength: Stockfish level {strength}")
```

## How to Run

### 1. Setup

1. **Install Dependencies**
   ```bash
   pip install torch python-chess numpy matplotlib
   ```

2. **Install Stockfish**
   - macOS: `brew install stockfish`
   - Ubuntu/Debian: `sudo apt-get install stockfish`
   - Windows: Download from [Stockfish website](https://stockfishchess.org/download/)

3. **Prepare Training Data**
   - Create a `data` directory
   - Add PGN files to the `data` directory
   - Recommended: Use high-quality games from [Lichess Elite Database](https://database.nikonoel.fr/)

### 2. Running from Console

1. **Create a Training Script**
   Create a file named `train.py` with the following content:
   ```python
   from train_model_stockfish import ChessTrainerWithStockfish
   
   def main():
       # Initialize trainer
       trainer = ChessTrainerWithStockfish(
           model_dir="models",
           stockfish_path="/usr/local/bin/stockfish",  # Update with your Stockfish path
           stockfish_level=8
       )
       
       # Train on PGN data first
       print("Starting PGN training phase...")
       trainer.train_from_pgn("data/", num_games=100)
       
       # Continue with self-play training
       print("\nStarting self-play training phase...")
       rewards, lengths, losses = trainer.train_self_play(num_episodes=1000)
   
   if __name__ == "__main__":
       main()
   ```

2. **Run the Training**
   ```bash
   # Make sure you're in the project directory
   cd /path/to/chess/training/project
   
   # Create necessary directories
   mkdir -p models data
   
   # Run the training script
   python train.py
   ```

3. **Monitor Training Progress**
   ```bash
   # View the latest training progress
   tail -f models/training_progress.log
   
   # Check model checkpoints
   ls -l models/checkpoint_*
   ```

4. **Test the Model**
   Create a file named `test_model.py`:
   ```python
   from train_model_stockfish import ChessTrainerWithStockfish
   
   def main():
       trainer = ChessTrainerWithStockfish(
           model_dir="models",
           stockfish_path="/usr/local/bin/stockfish"
       )
       strength = trainer.test_model_strength(num_games=10)
       print(f"Model strength: Stockfish level {strength}")
   
   if __name__ == "__main__":
       main()
   ```
   
   Run the test:
   ```bash
   python test_model.py
   ```

5. **Play Against the Model**
   Create a file named `play_chess.py`:
   ```python
   from train_model_stockfish import ChessAgent
   import chess
   
   def main():
       agent = ChessAgent(model_path="models/model_improved_final.pt")
       board = chess.Board()
       
       while not board.is_game_over():
           print("\nCurrent position:")
           print(board)
           
           if board.turn == chess.WHITE:
               move = input("Enter your move (e.g., e2e4): ")
               try:
                   board.push(chess.Move.from_uci(move))
               except:
                   print("Invalid move! Try again.")
                   continue
           else:
               move = agent.select_move(board)
               board.push(move)
               print(f"Model played: {move.uci()}")
       
       print("\nGame Over!")
       print(f"Result: {board.outcome().result()}")
   
   if __name__ == "__main__":
       main()
   ```
   
   Run the game:
   ```bash
   python play_chess.py
   ```

6. **Common Console Commands**
   ```bash
   # Check GPU availability (if using CUDA)
   nvidia-smi
   
   # Monitor system resources during training
   top
   
   # Check training logs
   cat models/training_progress.log
   
   # Backup model checkpoints
   cp -r models models_backup_$(date +%Y%m%d)
   ```

### 3. Training Parameters

You can adjust these parameters in `train_model_stockfish.py`:

```python
# Training parameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
TARGET_UPDATE = 50
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000

# Curriculum stages
CURRICULUM_STAGES = [
    {"max_moves": 20, "opponent_strength": 0.2},  # Opening
    {"max_moves": 40, "opponent_strength": 0.4},  # Middlegame
    {"max_moves": 60, "opponent_strength": 0.6},  # Endgame
    {"max_moves": 100, "opponent_strength": 0.8}, # Full game
]
```

### 4. Expected Training Time

- PGN Training: 1-2 hours (100 games)
- Self-Play Training: 5-10 hours (1000 episodes)
- Total: 6-12 hours

### 5. Monitoring Progress

1. **Console Output**
   ```
   Episode 100/1000 | Avg Reward: 0.45 | Avg Length: 35.2 | Avg Loss: 0.0234
   Current model strength: Stockfish level 4
   Learning Rate: 0.000800
   ```

2. **Training Plots**
   - Check `models/training_progress.png`
   - Shows rewards, lengths, losses, and Q-values

3. **Model Checkpoints**
   - Saved every 100 episodes
   - Located in `models/` directory
   - Format: `checkpoint_episode_X.pt`

### 6. Troubleshooting Common Issues

1. **Out of Memory**
   ```python
   # Reduce batch size and memory size
   BATCH_SIZE = 512
   MEMORY_SIZE = 50000
   ```

2. **Slow Training**
   ```python
   # Enable GPU if available
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

3. **Poor Performance**
   ```python
   # Adjust Stockfish level
   trainer = ChessTrainerWithStockfish(stockfish_level=6)  # Start with lower level
   ```

## Performance Expectations

1. **Initial Stage**
   - Should reach Stockfish level 3-4
   - Basic tactical understanding
   - Opening principles

2. **Mid Training**
   - Should reach Stockfish level 6-7
   - Good tactical play
   - Basic strategic understanding

3. **Final Stage**
   - Should reach Stockfish level 8-9
   - Strong tactical play
   - Good strategic understanding
   - Solid endgame play

## Requirements

- Python 3.7+
- PyTorch
- python-chess
- Stockfish 15+
- CUDA (optional, for GPU acceleration)

## Best Practices

1. **Training Duration**
   - Minimum: 1000 episodes
   - Recommended: 5000+ episodes
   - Check strength every 100 episodes

2. **Resource Management**
   - Use GPU if available
   - Monitor memory usage
   - Regular checkpointing

3. **Evaluation**
   - Test against multiple Stockfish levels
   - Track win rates
   - Monitor learning progress

## Troubleshooting

1. **Slow Learning**
   - Increase batch size
   - Adjust learning rate
   - Check reward scaling

2. **Poor Performance**
   - Verify Stockfish configuration
   - Check reward function
   - Adjust curriculum stages

3. **Memory Issues**
   - Reduce replay memory size
   - Decrease batch size
   - Use gradient checkpointing

## Future Improvements

1. **Architecture**
   - Transformer-based model
   - Multi-head attention
   - Policy gradient methods

2. **Training**
   - Parallel self-play
   - Distributed training
   - Advanced curriculum

3. **Evaluation**
   - Tournament-style testing
   - Human player evaluation
   - Opening book integration
