# Changelog

All notable changes to the Chess Reinforcement Learning project will be documented in this file.

## [Unreleased]

### Added
- Feature to continue training from a saved model checkpoint
- New command-line argument `--load_model` to specify a model file to load before training
- Enhanced documentation for model loading functionality
- Improved error handling for model loading

## [2023-06-15] - Continue Training Feature

### Added
- Added a new command-line argument `--load_model` to specify a model file to load before training
- Enhanced the `load_model` method in `ChessTrainer` class to:
  - Handle both relative and absolute paths
  - Properly map the model to the correct device (CPU/GPU)
  - Include better error handling
  - Ensure both policy and target networks are updated and on the correct device
- Updated training configuration output to display loaded model information
- Added comprehensive documentation for the model loading feature

### Changed
- Modified the training flow to load the model if specified before starting training
- Updated main script documentation to include the new feature

### Technical Details
- The `load_model` method now uses `torch.load` with `map_location` parameter to ensure models load correctly on any device
- Added explicit calls to move models to the correct device after loading
- Added try/except block to handle potential errors during model loading
- Path handling now supports both relative paths (within the models directory) and absolute paths

## How to Use the Continue Training Feature

To continue training from a saved model file, use the `--load_model` parameter when running the training script:

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

## Benefits of This Feature

1. **Resume interrupted training**: If training was interrupted, you can continue from the last checkpoint.
2. **Iterative training**: Train on different datasets sequentially.
3. **Fine-tuning**: Start with a pre-trained model and fine-tune it on specific data.
4. **Experimentation**: Try different training strategies on the same base model.
