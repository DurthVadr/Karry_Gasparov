# Improved Chess Model

This directory contains an improved implementation of the chess model with enhanced training methods. The improvements address several limitations in the original implementation and should result in significantly better playing strength.

## Key Improvements

### 1. Prioritized Experience Replay
- **Prioritized Sampling**: More important experiences are sampled more frequently
- **Larger Memory Capacity**: Increased from 10,000 to 100,000 positions
- **Importance Sampling**: Corrects bias introduced by prioritized sampling

### 2. Enhanced Reward Function
- **Deeper Stockfish Analysis**: Increased depth from 5 to 12 for more accurate evaluation
- **Nuanced Reward Scaling**: Different scaling based on magnitude of position change
- **Chess Principles Rewards**: Additional rewards for good chess principles (center control, development, king safety)

### 3. Curriculum Learning in Self-Play
- **Gradually Increasing Opponent Strength**: Starts with a weak opponent and gradually increases difficulty
- **Position Repetition Detection**: Penalizes repetitive play to encourage diverse strategies
- **Best Model Tracking**: Saves the best performing model during training

### 4. Improved Training Process
- **Learning Rate Scheduling**: Gradually decreases learning rate for better convergence
- **Gradient Clipping**: Prevents exploding gradients for more stable training
- **Better Exploration Strategy**: More sophisticated exploration for discovering better moves

## How to Use

### Training the Improved Model

To train the improved model, use the `train_model_stockfish.py` script:

```bash
python train_model_stockfish.py
```

The script will:
1. Train on PGN files in the `data` directory
2. Continue with self-play training
3. Save models at regular intervals

### Playing Against the Improved Model

To play against the trained model in the chess GUI:

1. Start the chess GUI:
```bash
python chess_gui.py
```

2. Click on "Load Model" and select one of the trained models (e.g., `models/model_pgn_4000.pt` or `models/best_model.pt`)

3. Play as White by making your moves, then click "AI Move" to have the model respond as Black

## Model Files

After training, several model files will be created:
- `model_pgn_checkpoint.pt`: Checkpoint model during PGN training
- `model_pgn_X.pt`: Model after training on X games (e.g., `model_pgn_4000.pt`)
- `model_pgn_final.pt`: Model after completing PGN training
- `model_stockfish_episode_X.pt`: Checkpoints during self-play training
- `model_stockfish_final.pt`: Final model after completing all training
- `best_model.pt`: Best performing model during training (recommended for play)

## Training Progress

Training progress is saved as a plot in the models directory (`training_progress.png`), showing:
- Episode rewards
- Episode lengths
- Training loss
- Average Q-values

## Implementation Details

The improvements are implemented in the following file:
- `train_model_stockfish.py`: Contains the enhanced training implementation with prioritized experience replay, improved reward function, curriculum learning, and other training enhancements

The model can be used with the existing chess GUI:
- `chess_gui.py`: Provides a graphical interface for playing against the trained model
- `model_integration.py`: Handles integration between the GUI and the trained model

## Troubleshooting

If you encounter issues:

1. **GPU Memory Issues**: Reduce batch size or use CPU training
2. **Stockfish Not Found**: Specify the correct path to Stockfish
3. **Training Too Slow**: Reduce the number of games or episodes
4. **Model Plays Poorly**: Try using the best model instead of the final model

## Future Improvements

Potential areas for further improvement:
- **Enhanced Neural Network Architecture**: Implement residual networks or transformer-based architecture for better pattern recognition
- **Monte Carlo Tree Search (MCTS)**: Combine neural networks with MCTS for stronger play
- **Multi-task Learning**: Train the model to predict multiple chess-related tasks simultaneously
- **Opening Book Integration**: Add opening book knowledge for better early game play
- **Endgame Tablebases**: Incorporate perfect endgame play using tablebases
