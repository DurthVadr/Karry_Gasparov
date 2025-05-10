# Advanced Self-Play Training for Chess Model

This document explains how to use the advanced self-play training system to achieve a chess model that plays at Stockfish level 7 strength.

## Overview

The self-play training system uses a combination of techniques to efficiently train a chess model:

1. **Curriculum Learning**: The model starts playing against Stockfish level 1 and gradually advances to higher levels as it improves
2. **Model Pool**: Previous versions of the model are saved and used as opponents for diverse training
3. **Adaptive Difficulty**: The system automatically adjusts the difficulty based on the model's performance
4. **Regular Evaluation**: The model is periodically evaluated against the target Stockfish level
5. **Efficient Batch Processing**: Experiences are collected and processed in batches for better GPU utilization
6. **Milestone Saving**: When the model reaches a new Stockfish level, it's saved as a milestone

## Training Process

The training process follows these steps:

1. **Initial Training with PGN Data**: The model learns basic chess principles from master-level games
2. **Self-Play Training**: The model improves through playing against Stockfish and previous versions of itself
3. **Automatic Advancement**: When the model achieves a 60% win rate against a Stockfish level, it advances to the next level
4. **Regular Evaluation**: Every 500 episodes, the model is evaluated against the target level (Stockfish 7)
5. **Milestone Achievement**: When the model achieves a 50% score against the target level, it's considered successful

## How to Train to Stockfish Level 7

### Prerequisites

- A GPU with at least 8GB of VRAM is recommended
- Stockfish chess engine installed
- PGN data for initial training (optional but recommended)

### Training Command

To train a model to Stockfish level 7:

```bash
python train_model_stockfish.py --self_play --self_play_episodes=5000 --target_level=7 --stockfish_path=/path/to/stockfish
```

If you want to start with PGN training first (recommended):

```bash
python train_model_stockfish.py --data_dir=data --num_games=10000 --self_play --self_play_episodes=5000 --target_level=7
```

### Training Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `--self_play_episodes` | Number of self-play episodes | 5000 |
| `--target_level` | Target Stockfish level | 7 |
| `--batch_size` | Batch size for optimization | 64-256 (depending on GPU) |
| `--save_interval` | How often to save models | 100-500 |
| `--eval_interval` | How often to evaluate against target | 500 |

### Expected Training Time

Training time depends on your hardware, but here are some estimates:

- **PGN Training**: 2-4 hours for 10,000 games on a good GPU
- **Self-Play to Level 3**: 4-8 hours
- **Self-Play to Level 5**: 12-24 hours
- **Self-Play to Level 7**: 24-48 hours

Total expected time: 2-3 days on a good GPU.

## Monitoring Progress

The training process provides several ways to monitor progress:

1. **Console Output**: Shows current level, win rates, rewards, and other metrics
2. **Training Progress Plot**: Updated periodically with rewards, losses, win rates, and other metrics
3. **Milestone Models**: Saved when the model reaches a new Stockfish level
4. **Evaluation Results**: Detailed results when the model is evaluated against the target level

## Tips for Faster Training

1. **Start with PGN Training**: Initialize the model with knowledge from master-level games
2. **Use a Powerful GPU**: Training speed scales almost linearly with GPU performance
3. **Optimize Batch Size**: Use the largest batch size that fits in your GPU memory
4. **Reduce Stockfish Evaluation Frequency**: The system already samples only 25% of positions for Stockfish evaluation
5. **Focus on Early Levels**: The model learns most efficiently at lower levels (1-4)

## Troubleshooting

### Common Issues

1. **Stockfish Path Error**:
   - If you see `Error initializing opponent Stockfish: 'exepath'`, make sure to provide the correct path to Stockfish
   - Always use the `--stockfish_path` parameter with the full path to your Stockfish executable
   - Example: `--stockfish_path=/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish` (Mac)
   - Example: `--stockfish_path=C:\\path\\to\\stockfish.exe` (Windows)

2. **Slow Training**:
   - Check GPU utilization (should be >70%)
   - Reduce Stockfish evaluation frequency
   - Increase batch size

3. **Model Not Improving**:
   - Ensure PGN data is high quality
   - Try longer training at each level (adjust win_rate_threshold)
   - Check reward function and learning rate

4. **Out of Memory Errors**:
   - Reduce batch size
   - Close other applications using GPU memory

## Advanced Configuration

For advanced users, you can modify these parameters in the code:

- `win_rate_threshold`: How high the win rate must be to advance to the next level (default: 0.6)
- `win_rate_window`: How many games to consider when calculating win rate (default: 50)
- Learning rate and scheduler parameters
- Model architecture (in the DQN class)

## Example Training Timeline

Here's a typical progression for training to Stockfish level 7:

1. **PGN Training**: Learn basic chess principles (10,000 games)
2. **Level 1**: ~500 episodes to achieve 60% win rate
3. **Level 2**: ~500 episodes to achieve 60% win rate
4. **Level 3**: ~750 episodes to achieve 60% win rate
5. **Level 4**: ~750 episodes to achieve 60% win rate
6. **Level 5**: ~1000 episodes to achieve 60% win rate
7. **Level 6**: ~1000 episodes to achieve 60% win rate
8. **Level 7**: ~500 episodes to achieve 50% score (target achieved)

Total: ~5000 episodes of self-play training
