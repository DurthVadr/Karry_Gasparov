# Model Loading Guide

This guide explains how to use the model loading feature to continue training from a saved checkpoint.

## Overview

The model loading feature allows you to:
- Resume training that was interrupted
- Continue training a model on new data
- Fine-tune a pre-trained model
- Experiment with different training strategies on the same base model

## Command-Line Usage

To continue training from a saved model file, use the `--load_model` parameter when running the training script:

```bash
python main.py --load_model <path_to_model_file> [other_options]
```

### Examples

1. **Continue PGN training from a saved model:**
   ```bash
   python main.py --data_dir data --load_model models/model_pgn_checkpoint.pt
   ```

2. **Continue self-play training from a saved model:**
   ```bash
   python main.py --self_play --load_model models/model_self_play_episode_1000.pt
   ```

3. **Load a model from a specific path (not in the models directory):**
   ```bash
   python main.py --data_dir data --load_model /path/to/your/model.pt
   ```

## Path Handling

The `--load_model` parameter accepts two types of paths:

1. **Relative paths**: If you provide just a filename or a relative path without a leading `/`, the system will look for the model in the `models` directory.
   ```bash
   python main.py --load_model model_pgn_checkpoint.pt
   ```

2. **Absolute paths**: If you provide a full path starting with `/`, the system will use that exact path.
   ```bash
   python main.py --load_model /home/user/chess_models/my_model.pt
   ```

## Device Compatibility

The model loading feature automatically handles device compatibility:

- If you trained on a GPU and are now loading on a CPU (or vice versa), the model will be correctly mapped to the available device.
- The system uses `torch.load` with the `map_location` parameter to ensure models load correctly on any device.

## Error Handling

If the model file cannot be found or loaded, the system will:
1. Print an error message
2. Continue training with a new model

Example error message:
```
Loading model from models/non_existent_model.pt to continue training...
Model file models/non_existent_model.pt not found
Failed to load model. Training will start with a new model.
```

## Programmatic Usage

You can also use the model loading feature programmatically:

```python
from main import ChessTrainer

# Create a trainer instance
trainer = ChessTrainer(stockfish_path="/path/to/stockfish")

# Load a model
if trainer.load_model("models/my_model.pt"):
    print("Model loaded successfully")
else:
    print("Failed to load model")

# Continue with training
trainer.train_from_pgn("data", num_games=1000)
```

## Common Use Cases

### 1. Resuming Interrupted Training

If your training was interrupted (e.g., due to a power outage or system crash), you can resume from the last saved checkpoint:

```bash
python main.py --data_dir data --load_model models/model_pgn_checkpoint.pt
```

### 2. Sequential Training on Different Datasets

Train on one dataset, then continue training on another:

```bash
# First training session
python main.py --data_dir data/dataset1 --save_interval 1000

# Continue training on a second dataset
python main.py --data_dir data/dataset2 --load_model models/model_pgn_final.pt
```

### 3. Fine-tuning a Pre-trained Model

Start with a general model and fine-tune it on specific data:

```bash
# Load a general model and fine-tune on specific data
python main.py --data_dir data/specific_positions --load_model models/general_model.pt
```

### 4. Experimenting with Different Training Strategies

Try different training strategies on the same base model:

```bash
# Train with self-play using different parameters
python main.py --self_play --self_play_episodes 2000 --load_model models/base_model.pt --target_level 5

# Try a different approach with the same base model
python main.py --self_play --self_play_episodes 3000 --load_model models/base_model.pt --target_level 7
```

## Troubleshooting

### Model Not Found

If you get a "Model file not found" error:
- Check that the path is correct
- Verify that the model file exists
- If using a relative path, make sure it's relative to the `models` directory

### CUDA Out of Memory

If you get a CUDA out of memory error after loading a model:
- The model might be too large for your GPU
- Try reducing the batch size: `--batch_size 32` (or smaller)
- Consider using CPU training if GPU memory is limited

### Model Incompatibility

If you get errors about incompatible model structure:
- The model might have been trained with a different version of the code
- Check that you're using the same network architecture
- Consider training a new model if the architectures are incompatible
