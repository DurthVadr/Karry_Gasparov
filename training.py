"""
Training methods for chess reinforcement learning.
"""

import os
import time
import random
import torch
import chess
import chess.pgn
import chess.engine
import numpy as np
from drl_agent import DQN, board_to_tensor, create_move_mask
from utils import plot_training_progress

class PGNTrainer:
    """
    Training from PGN files with optimized batch processing.
    """

    def __init__(self, trainer):
        """
        Initialize the PGN trainer.

        Args:
            trainer: ChessTrainer instance
        """
        self.trainer = trainer

    def train_from_pgn(self, pgn_path, num_games=1000, batch_size=32, save_interval=100):
        """
        Train the model from PGN games with optimized batch processing.

        This improved version uses batch processing and efficient tensor operations
        to significantly speed up training from PGN files.

        Args:
            pgn_path: Path to a PGN file or directory containing PGN files
            num_games: Maximum number of games to process
            batch_size: Number of positions to process in each batch
            save_interval: How often to save the model (in games)

        Returns:
            List of losses from training
        """
        game_count = 0
        losses = []
        start_time = time.time()
        total_positions = 0

        # Check if pgn_path is a directory
        if os.path.isdir(pgn_path):
            print(f"Training from PGN files in directory: {pgn_path}")
            # Get all PGN files in the directory
            pgn_files = [os.path.join(pgn_path, f) for f in os.listdir(pgn_path)
                        if f.endswith('.pgn') and os.path.isfile(os.path.join(pgn_path, f))]

            if not pgn_files:
                print(f"No PGN files found in directory: {pgn_path}")
                return losses

            print(f"Found {len(pgn_files)} PGN files")

            # Process each PGN file
            for pgn_file in sorted(pgn_files):
                if game_count >= num_games:
                    break

                print(f"Processing file: {os.path.basename(pgn_file)}")
                file_losses, file_games, file_positions = self._process_pgn_file(
                    pgn_file, game_count, num_games - game_count, batch_size, save_interval
                )

                losses.extend(file_losses)
                game_count += file_games
                total_positions += file_positions

                # Save model after each file
                self.trainer.save_model(f"model_pgn_checkpoint.pt")

                # Print statistics
                elapsed_time = time.time() - start_time
                positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0
                print(f"Progress: {game_count}/{num_games} games | {total_positions} positions | "
                      f"{positions_per_second:.1f} positions/sec | "
                      f"Time elapsed: {elapsed_time:.1f}s")
        else:
            # Process a single PGN file
            print(f"Training from PGN file: {pgn_path}")
            file_losses, file_games, file_positions = self._process_pgn_file(
                pgn_path, 0, num_games, batch_size, save_interval
            )

            losses.extend(file_losses)
            game_count = file_games
            total_positions = file_positions

        # Save final model
        self.trainer.save_model("model_pgn_final.pt")

        # Print final statistics
        elapsed_time = time.time() - start_time
        positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0
        print(f"PGN training completed! Processed {game_count} games, {total_positions} positions")
        print(f"Average speed: {positions_per_second:.1f} positions/sec")
        print(f"Total training time: {elapsed_time:.1f} seconds ({elapsed_time/3600:.2f} hours)")

        # Generate and save metrics plots
        metrics_dir = os.path.join(self.trainer.model_dir, 'metrics')
        self.trainer.plot_metrics(output_dir=metrics_dir)

        return losses

    def _process_pgn_file(self, pgn_file, start_count, max_games, batch_size=32, save_interval=100):
        """
        Process a single PGN file for training with optimized batch processing.

        This improved version:
        1. Collects positions in batches before processing
        2. Uses vectorized operations where possible
        3. Performs fewer Stockfish evaluations by sampling
        4. Optimizes memory usage and tensor operations

        Args:
            pgn_file: Path to the PGN file
            start_count: Starting game count
            max_games: Maximum number of games to process
            batch_size: Number of positions to process in each batch
            save_interval: How often to save the model (in games)

        Returns:
            Tuple of (losses, game_count, position_count)
        """
        losses = []
        game_count = 0
        position_count = 0
        optimization_steps = 0

        # Track time for performance monitoring
        start_time = time.time()
        last_report_time = start_time

        try:
            with open(pgn_file) as f:
                while game_count < max_games:
                    # Read the next game
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break  # End of file

                    # Process the game
                    board = game.board()
                    moves = list(game.mainline_moves())

                    # Skip very short games
                    if len(moves) < 10:
                        continue

                    # Process positions in the game
                    positions = []

                    # First, collect all positions from the game
                    for i, move in enumerate(moves):
                        # Store current board and move
                        current_board = board.copy()

                        # Convert move to action index
                        action_idx = move.from_square * 64 + move.to_square

                        # Make the move
                        board.push(move)

                        # Check if game is over
                        done = board.is_game_over()

                        # Store position data
                        positions.append((current_board, action_idx, board.copy(), done))

                    # Process positions in batches for efficiency
                    for i in range(0, len(positions), batch_size):
                        batch_positions = positions[i:i+batch_size]

                        # Process this batch
                        current_boards = [pos[0] for pos in batch_positions]
                        action_indices = [pos[1] for pos in batch_positions]
                        next_boards = [pos[2] for pos in batch_positions]
                        dones = [pos[3] for pos in batch_positions]

                        # Calculate rewards in batch with adaptive sampling
                        rewards = []
                        for j, (curr_board, next_board) in enumerate(zip(current_boards, next_boards)):
                            # Use adaptive sampling based on position importance
                            # The reward calculator will determine importance internally
                            # and adjust evaluation depth accordingly
                            reward = self.trainer.reward_calculator.calculate_stockfish_reward(next_board, curr_board)
                            rewards.append(reward)

                        # Convert boards to tensors efficiently (in batch)
                        states = []
                        masks = []
                        next_states = []
                        next_masks = []

                        for curr_board, next_board in zip(current_boards, next_boards):
                            # Convert to tensors and move to device in one operation
                            states.append(board_to_tensor(curr_board, self.trainer.device))
                            masks.append(create_move_mask(curr_board, self.trainer.device))
                            next_states.append(board_to_tensor(next_board, self.trainer.device))
                            next_masks.append(create_move_mask(next_board, self.trainer.device))

                        # Store experiences in memory
                        for state, action_idx, next_state, reward, done, mask, next_mask in zip(
                            states, action_indices, next_states, rewards, dones, masks, next_masks
                        ):
                            self.trainer.memory.push(state, action_idx, next_state, reward, done, mask, next_mask)
                            position_count += 1

                        # Perform optimization if enough samples
                        if len(self.trainer.memory) >= self.trainer.batch_size:
                            loss = self.trainer.optimize_model()
                            if loss is not None:
                                losses.append(loss)
                                optimization_steps += 1

                    game_count += 1

                    # Update target network periodically based on optimization steps
                    if optimization_steps > 0 and optimization_steps % self.trainer.target_update == 0:
                        self.trainer.target_net.load_state_dict(self.trainer.policy_net.state_dict())

                    # Print progress every 10 seconds
                    current_time = time.time()
                    if current_time - last_report_time > 10:
                        elapsed_time = current_time - start_time
                        positions_per_second = position_count / elapsed_time if elapsed_time > 0 else 0
                        avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1) if losses else 0

                        print(f"Processed {start_count + game_count} games | {position_count} positions | "
                              f"{positions_per_second:.1f} pos/sec | Avg Loss: {avg_loss:.4f}")

                        # Display simple metrics
                        self.trainer.metrics.print_metrics()

                        last_report_time = current_time

                    # Save model periodically
                    if game_count % save_interval == 0:
                        checkpoint_path = f"model_pgn_{start_count + game_count}.pt"
                        self.trainer.save_model(checkpoint_path)

                        # Calculate average loss for scheduler
                        avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1) if losses else 0

                        # Apply learning rate scheduling with loss value for ReduceLROnPlateau
                        self.trainer.scheduler.step(avg_loss)

                        # Get current learning rate
                        current_lr = self.trainer.optimizer.param_groups[0]['lr']
                        print(f"Current learning rate: {current_lr:.6f}")

                        # Print current training metrics
                        print("\n=== Current Training Metrics ===")
                        self.trainer.print_training_metrics()

                        # Evaluate the checkpoint model if Stockfish is available
                        if self.trainer.stockfish_path:
                            print(f"\n=== Evaluating checkpoint model: {checkpoint_path} ===")
                            # Use a smaller number of games for faster evaluation during training
                            results = self.trainer.model_evaluator.evaluate_against_stockfish(
                                model_path=os.path.join(self.trainer.model_dir, checkpoint_path),
                                num_games=3,  # Fewer games for quicker evaluation
                                stockfish_levels=[1, 3, 5]  # Test against a few key levels
                            )

                            # Print brief summary
                            if results:
                                # Determine approximate Stockfish level of the model
                                best_comparable_level = 0
                                for level in sorted(results.keys()):
                                    if results[level]['score'] >= 0.45:  # Model is competitive (45%+ score)
                                        best_comparable_level = level

                                strength_msg = f"Stockfish level {best_comparable_level}" if best_comparable_level > 0 else "below Stockfish level 1"
                                print(f"\nCheckpoint Evaluation: Model strength ~ {strength_msg}")

        except Exception as e:
            print(f"Error processing file {pgn_file}: {e}")
            import traceback
            traceback.print_exc()

        return losses, game_count, position_count


class SelfPlayTrainer:
    """
    Advanced self-play training with enhanced curriculum learning.

    This trainer implements:
    - Adaptive curriculum learning based on win rate, draw rate, and game quality
    - Non-linear difficulty progression for more natural learning
    - Phase-aware exploration strategies
    - Diverse opponent pool with strategic selection
    - Comprehensive evaluation metrics
    """

    def __init__(self, trainer):
        """
        Initialize the self-play trainer.

        Args:
            trainer: ChessTrainer instance
        """
        self.trainer = trainer

        # Curriculum learning parameters
        self.curriculum_params = {
            'win_threshold': 0.52,       # Base win rate threshold for advancing (slightly lower for faster progression)
            'draw_threshold': 0.30,      # Draw rate threshold for consideration (higher to value draws more)
            'quality_threshold': 0.55,   # Game quality threshold (slightly lower for faster progression)
            'window_size': 30,           # Number of games to calculate metrics over (smaller window for faster adaptation)
            'difficulty_scale': [        # Non-linear difficulty scale
                {'level': 1, 'factor': 1.0},
                {'level': 2, 'factor': 1.2},
                {'level': 3, 'factor': 1.5},
                {'level': 4, 'factor': 1.8},
                {'level': 5, 'factor': 2.2},
                {'level': 6, 'factor': 2.7},
                {'level': 7, 'factor': 3.3},
                {'level': 8, 'factor': 4.0},
                {'level': 9, 'factor': 5.0},
                {'level': 10, 'factor': 6.0}
            ],
            'sublevel_steps': 3,         # Fewer sublevels for faster level progression
            'regression_threshold': 0.25  # Lower threshold to avoid too much regression
        }

    def _calculate_effective_level(self, level, sublevel):
        """
        Calculate the effective difficulty level based on current level and sublevel.

        This creates a smoother progression between integer difficulty levels.

        Args:
            level (int): Current curriculum level
            sublevel (int): Current sublevel within the level

        Returns:
            float: Effective difficulty level
        """
        # Calculate fractional level for smoother progression
        sublevel_fraction = (sublevel - 1) / self.curriculum_params['sublevel_steps']
        return level + sublevel_fraction

    def _update_opponent_weights(self, weights, level):
        """
        Update opponent selection weights based on curriculum level.

        Args:
            weights (dict): Dictionary of opponent weights to update
            level (int): Current curriculum level
        """
        # Adjust weights based on level
        if level <= 2:
            # More random opponents at early levels for exploration
            weights['random'] = 0.3
            weights['self'] = 0.5
            weights['pool'] = 0.2
        elif level <= 5:
            # Balanced mix in middle levels
            weights['random'] = 0.1
            weights['self'] = 0.4
            weights['pool'] = 0.5
        else:
            # More pool opponents at higher levels for challenge
            weights['random'] = 0.05
            weights['self'] = 0.25
            weights['pool'] = 0.7

    def _select_opponent_type(self, weights):
        """
        Select opponent type using weighted random selection.

        Args:
            weights (dict): Dictionary of opponent weights

        Returns:
            str: Selected opponent type ('self', 'pool', or 'random')
        """
        # Normalize weights
        total = sum(weights.values())
        normalized_weights = {k: v/total for k, v in weights.items()}

        # Weighted random selection
        r = random.random()
        cumulative = 0
        for opponent_type, weight in normalized_weights.items():
            cumulative += weight
            if r <= cumulative:
                return opponent_type

        # Default to self-play if something goes wrong
        return 'self'

    def _select_appropriate_pool_model(self, model_pool, metadata, target_level):
        """
        Select an appropriate model from the pool based on curriculum level.

        Args:
            model_pool (list): List of models in the pool
            metadata (list): List of model metadata
            target_level (float): Target difficulty level

        Returns:
            nn.Module: Selected model from the pool
        """
        if not model_pool:
            raise ValueError("Model pool is empty")

        # Find models close to the target level
        suitable_models = []
        for i, _ in enumerate(model_pool):
            # If model has a level attribute, use it
            model_level = metadata[i].get('level', 1)

            # Calculate distance from target level
            level_distance = abs(model_level - target_level)

            # Consider models within reasonable distance
            if level_distance <= 2:
                suitable_models.append((i, level_distance))

        # If no suitable models found, use any model
        if not suitable_models:
            return random.choice(model_pool)

        # Sort by distance (closest first)
        suitable_models.sort(key=lambda x: x[1])

        # Select from the closest models with some randomness
        selection_pool = suitable_models[:min(3, len(suitable_models))]
        selected_idx = random.choice(selection_pool)[0]

        return model_pool[selected_idx]

    def _evaluate_move_quality(self, board, move, stockfish, depth=12):
        """
        Evaluate the quality of a move compared to Stockfish's best move.

        Args:
            board (chess.Board): Board position before the move
            move (chess.Move): The move to evaluate
            stockfish (chess.engine.SimpleEngine): Stockfish engine
            depth (int): Evaluation depth

        Returns:
            float: Move quality score (0-1)
        """
        try:
            # Get Stockfish's evaluation of the position
            result = stockfish.analyse(board, chess.engine.Limit(depth=depth))
            stockfish_best_move = result.get("pv")[0]
            stockfish_score = result.get("score").relative.score(mate_score=10000)

            # Make the move to evaluate
            board_copy = board.copy()
            board_copy.push(move)

            # Get Stockfish's evaluation after the move
            result_after = stockfish.analyse(board_copy, chess.engine.Limit(depth=depth))
            move_score = -result_after.get("score").relative.score(mate_score=10000)

            # Calculate the difference between the best move and the played move
            if stockfish_best_move == move:
                return 1.0  # Perfect move

            # Calculate score difference
            score_diff = abs(stockfish_score - move_score)

            # Convert score difference to quality (0-1)
            # Small differences (< 50 centipawns) are still high quality
            if score_diff < 50:
                quality = 1.0 - (score_diff / 100)
            # Medium differences (50-200 centipawns) are medium quality
            elif score_diff < 200:
                quality = 0.75 - ((score_diff - 50) / 600)
            # Large differences (> 200 centipawns) are low quality
            else:
                quality = max(0.1, 0.5 - ((score_diff - 200) / 1600))

            return max(0.0, min(1.0, quality))
        except Exception as e:
            # If evaluation fails, return a default medium quality
            print(f"Move quality evaluation failed: {e}")
            return 0.5

    def _get_difficulty_factor(self, level):
        """
        Get the difficulty scaling factor for the current level.

        Args:
            level (int): Current curriculum level

        Returns:
            float: Difficulty scaling factor
        """
        # Find the difficulty factor from the scale
        for entry in self.curriculum_params['difficulty_scale']:
            if entry['level'] == level:
                return entry['factor']

        # Default to the factor for the closest level
        closest_level = min(self.curriculum_params['difficulty_scale'],
                           key=lambda x: abs(x['level'] - level))
        return closest_level['factor']

    def train_self_play(self, num_episodes=5000, stockfish_levels=None,
                      batch_size=64, save_interval=100, eval_interval=1000, target_level=7):
        """
        Enhanced self-play training with advanced curriculum learning.

        This optimized version includes:
        1. Self-play against current model (model vs itself)
        2. Self-play against previous model versions from the model pool
        3. Batch processing of experiences for efficient training
        4. Periodic evaluation against Stockfish for benchmarking
        5. Comprehensive checkpointing for continuous training
        6. Advanced curriculum learning with non-linear difficulty progression
        7. Game quality assessment for curriculum advancement
        8. Draw rate consideration in addition to win rate

        Args:
            num_episodes: Total number of episodes to train
            stockfish_levels: List of Stockfish levels to evaluate against (default: [5,6,7])
            batch_size: Batch size for optimization
            save_interval: How often to save models (in episodes)
            eval_interval: How often to evaluate against Stockfish (every 1000 games)
            target_level: Target Stockfish level to achieve

        Returns:
            Tuple of (episode_rewards, episode_lengths, losses)
        """
        print(f"Starting enhanced self-play training with advanced curriculum learning for {num_episodes} episodes...")
        print(f"Target performance: Stockfish level {target_level}")
        print(f"Evaluation interval: Every {eval_interval} games")
        print(f"Using non-linear difficulty progression with {self.curriculum_params['sublevel_steps']} sublevels per level")
        print(f"Considering win rate, draw rate, and game quality for curriculum advancement")

        # Set default Stockfish levels for evaluation if not provided
        if stockfish_levels is None:
            stockfish_levels = [5, 6, 7]  # Focus on target levels for evaluation

        # Initialize tracking metrics
        self.trainer.training_stats['episode_rewards'] = []
        self.trainer.training_stats['episode_lengths'] = []
        self.trainer.training_stats['losses'] = []
        self.trainer.training_stats['win_rates'] = []
        self.trainer.training_stats['draw_rates'] = []
        self.trainer.training_stats['game_quality'] = []
        self.trainer.training_stats['curriculum_level'] = []

        # Initialize model pool for diverse opponents
        model_pool = []
        model_pool_metadata = []
        model_dir = self.trainer.model_dir

        # Initialize curriculum tracking
        self.current_level = 1
        self.current_sublevel = 1
        self.games_at_current_level = 0
        self.wins_at_current_level = 0
        self.draws_at_current_level = 0
        self.quality_sum_at_current_level = 0

        # Game outcome history for calculating metrics
        self.game_outcomes = []  # List of 'win', 'loss', 'draw'
        self.game_qualities = [] # List of game quality scores (0-1)

        # Check if we have existing models in the model directory to add to the pool
        existing_models = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'model_pool' in f]
        if existing_models:
            print(f"Found {len(existing_models)} existing models to add to the model pool")
            max_pool_size = self.trainer.hyperparams['curriculum']['max_pool_size']
            for model_file in existing_models[:max_pool_size]:  # Limit to max pool size
                try:
                    model_path = os.path.join(model_dir, model_file)
                    model_copy = DQN()
                    model_copy.load_state_dict(torch.load(model_path, map_location=self.trainer.device))
                    
                    episode_num = int(model_file.split('_')[-1].split('.')[0])

                    # Only append if all above succeeds
                    model_pool.append(model_copy)
                    model_pool_metadata.append({
                        'episode': episode_num,
                        'path': model_path,
                        'level': 1  # Default level
                    })
                    print(f"Added {model_file} to model pool")

                except Exception as e:
                    print(f"Error loading model {model_file}: {e}")


            print(f"Successfully loaded {len(model_pool)} models into the pool")

        best_model_path = os.path.join(model_dir, "best_model.pt")

        # Track training progress
        start_time = time.time()
        total_positions = 0
        total_optimization_steps = 0

        # Enhanced model pool parameters
        max_pool_size = 10  # Size of model pool for diverse opponents
        model_pool_metadata = []  # Track metadata about each model in the pool

        # Opponent selection weights - will be adjusted based on curriculum level
        self.opponent_weights = {
            'self': 0.3,      # Play against itself
            'pool': 0.5,      # Play against model pool
            'random': 0.2     # Play against random moves (for exploration)
        }

        # Create a separate Stockfish instance for evaluation only
        evaluation_stockfish = None
        if self.trainer.stockfish_path:
            try:
                # Initialize Stockfish for evaluation only
                evaluation_stockfish = chess.engine.SimpleEngine.popen_uci(self.trainer.stockfish_path)
                print(f"Initialized Stockfish for evaluation using path: {self.trainer.stockfish_path}")
            except Exception as e:
                print(f"Error initializing Stockfish for evaluation: {e}")
                print("Stockfish evaluation will not be available")

        # Main training loop
        for episode in range(num_episodes):
            # Calculate effective difficulty level based on current level and sublevel
            effective_level = self._calculate_effective_level(self.current_level, self.current_sublevel)

            # Update opponent selection weights based on curriculum level
            self._update_opponent_weights(self.opponent_weights, self.current_level)

            # Select opponent type using weighted random selection
            opponent_type = self._select_opponent_type(self.opponent_weights)

            # Initialize opponent model based on selected type
            opponent_model = None
            use_model_pool = False
            use_random_moves = False

            if opponent_type == 'pool' and len(model_pool) > 0:
                # Select a model from the pool based on curriculum level
                opponent_model = self._select_appropriate_pool_model(model_pool, model_pool_metadata, effective_level)
                opponent_model.eval()  # Set to evaluation mode
                opponent_model.to(self.trainer.device)
                use_model_pool = True
            elif opponent_type == 'random':
                use_random_moves = True
            # else: use self-play (opponent_type == 'self')

            # Initialize the game
            board = chess.Board()
            episode_reward = 0
            episode_length = 0
            positions_seen = {}  # Track positions for repetition detection
            experiences = []  # Collect experiences for batch processing
            move_qualities = []  # Track quality of moves for game quality assessment

            # Play the game
            done = False
            while not done:
                # Get current state
                state = board_to_tensor(board, self.trainer.device)
                mask = create_move_mask(board, self.trainer.device)

                if board.turn == chess.WHITE:  # Our model plays as White
                    # Select action using our policy with phase-aware exploration
                    action_idx, move = self.trainer.select_action(state, mask, board)

                    # Evaluate move quality if Stockfish is available (for White only)
                    if evaluation_stockfish:
                        move_quality = self._evaluate_move_quality(board, move, evaluation_stockfish)
                        move_qualities.append(move_quality)
                else:  # Opponent plays as Black
                    if use_model_pool:
                        # Get move from opponent model in the pool
                        with torch.no_grad():
                            q_values = opponent_model(state, mask)
                            legal_moves = list(board.legal_moves)
                            legal_move_indices = [m.from_square * 64 + m.to_square for m in legal_moves]
                            legal_q_values = q_values[0, legal_move_indices]
                            best_idx = torch.argmax(legal_q_values).item()
                            move = legal_moves[best_idx]
                            action_idx = legal_move_indices[best_idx]
                    elif use_random_moves:
                        # Choose a random legal move
                        legal_moves = list(board.legal_moves)
                        move = random.choice(legal_moves)
                        action_idx = move.from_square * 64 + move.to_square
                    else:
                        # Self-play against current policy (model plays against itself)
                        action_idx, move = self.trainer.select_action(state, mask, board)

                # Store current board for reward calculation
                prev_board = board.copy()

                # Execute the move
                board.push(move)

                # Get next state
                next_state = board_to_tensor(board, self.trainer.device)
                next_mask = create_move_mask(board, self.trainer.device)

                # Check if game is over
                done = board.is_game_over()

                # Calculate reward using adaptive sampling with difficulty adjustment
                base_reward = self.trainer.reward_calculator.calculate_stockfish_reward(board, prev_board)

                # Scale reward based on curriculum level for better learning progression
                reward = base_reward * self._get_difficulty_factor(self.current_level)

                # Handle repetitions with enhanced penalties
                board_fen = board.fen().split(' ')[0]
                if board_fen in positions_seen:
                    positions_seen[board_fen] += 1
                    if positions_seen[board_fen] == 2:
                        reward -= 0.5  # Mild penalty for first repetition
                    elif positions_seen[board_fen] >= 3:
                        reward -= 2.0  # Severe penalty for threefold repetition
                        done = True
                else:
                    positions_seen[board_fen] = 1

                # Store experience for batch processing
                experiences.append((state, action_idx, next_state, reward, done, mask, next_mask))

                episode_reward += reward
                episode_length += 1
                total_positions += 1

                # End game if it's too long
                if episode_length >= 300:
                    done = True

            # Process experiences in batches
            for i in range(0, len(experiences), batch_size):
                batch = experiences[i:min(i+batch_size, len(experiences))]

                # Add experiences to memory
                for exp in batch:
                    self.trainer.memory.push(*exp)

                # Perform optimization if enough samples
                if len(self.trainer.memory) >= self.trainer.batch_size:
                    loss = self.trainer.optimize_model()
                    if loss is not None:
                        self.trainer.training_stats['losses'].append(loss)
                        total_optimization_steps += 1

            # Update target network periodically
            if total_optimization_steps > 0 and total_optimization_steps % self.trainer.target_update == 0:
                self.trainer.target_net.load_state_dict(self.trainer.policy_net.state_dict())
                print(f"Target network updated after {total_optimization_steps} optimization steps")

            # Track metrics
            self.trainer.training_stats['episode_rewards'].append(episode_reward)
            self.trainer.training_stats['episode_lengths'].append(episode_length)
            self.trainer.training_stats['curriculum_level'].append(effective_level)

            # Calculate game quality (average move quality)
            game_quality = sum(move_qualities) / max(len(move_qualities), 1) if move_qualities else 0.5
            self.game_qualities.append(game_quality)

            # Track game outcome for statistics and curriculum advancement
            game_outcome = None
            if board.is_checkmate():
                if board.turn == chess.BLACK:  # Model (White) won
                    game_outcome = "win"
                else:  # Model (Black) lost
                    game_outcome = "loss"
            elif board.is_stalemate() or board.is_insufficient_material():
                game_outcome = "draw"
            else:
                game_outcome = "incomplete"

            # Update game outcomes history
            self.game_outcomes.append(game_outcome)

            # Update curriculum tracking
            self.games_at_current_level += 1
            if game_outcome == "win":
                self.wins_at_current_level += 1
            elif game_outcome == "draw":
                self.draws_at_current_level += 1
            self.quality_sum_at_current_level += game_quality

            # Check if we should update curriculum level
            if self.games_at_current_level >= self.curriculum_params['window_size']:
                # Calculate metrics
                win_rate = self.wins_at_current_level / self.games_at_current_level
                draw_rate = self.draws_at_current_level / self.games_at_current_level
                avg_quality = self.quality_sum_at_current_level / self.games_at_current_level

                # Store metrics
                self.trainer.training_stats['win_rates'].append(win_rate)
                self.trainer.training_stats['draw_rates'].append(draw_rate)
                self.trainer.training_stats['game_quality'].append(avg_quality)

                # Determine if we should advance, maintain, or regress the curriculum
                should_advance = (
                    win_rate >= self.curriculum_params['win_threshold'] and
                    avg_quality >= self.curriculum_params['quality_threshold']
                )

                # Consider draws as partial success for advancement
                if not should_advance and draw_rate > self.curriculum_params['draw_threshold']:
                    # If many draws but good quality, still consider advancing
                    combined_success_rate = win_rate + (draw_rate * 0.5)
                    should_advance = (
                        combined_success_rate >= self.curriculum_params['win_threshold'] and
                        avg_quality >= self.curriculum_params['quality_threshold']
                    )

                # Check if we should regress due to poor performance
                should_regress = win_rate < self.curriculum_params['regression_threshold']

                if should_advance:
                    # Advance to next sublevel
                    self.current_sublevel += 1
                    if self.current_sublevel > self.curriculum_params['sublevel_steps']:
                        # Advance to next level
                        self.current_sublevel = 1
                        self.current_level += 1
                        print(f"Curriculum advanced to level {self.current_level}.{self.current_sublevel} "
                              f"(win rate: {win_rate:.2f}, draw rate: {draw_rate:.2f}, quality: {avg_quality:.2f})")
                    else:
                        print(f"Curriculum advanced to sublevel {self.current_level}.{self.current_sublevel} "
                              f"(win rate: {win_rate:.2f}, draw rate: {draw_rate:.2f}, quality: {avg_quality:.2f})")
                elif should_regress:
                    # Regress to previous sublevel
                    self.current_sublevel -= 1
                    if self.current_sublevel < 1:
                        # Regress to previous level
                        self.current_level = max(1, self.current_level - 1)
                        self.current_sublevel = self.curriculum_params['sublevel_steps']
                        print(f"Curriculum regressed to level {self.current_level}.{self.current_sublevel} "
                              f"due to poor performance (win rate: {win_rate:.2f})")
                    else:
                        print(f"Curriculum regressed to sublevel {self.current_level}.{self.current_sublevel} "
                              f"(win rate: {win_rate:.2f})")

                # Reset tracking for next window
                self.games_at_current_level = 0
                self.wins_at_current_level = 0
                self.draws_at_current_level = 0
                self.quality_sum_at_current_level = 0

            # Log game outcome for statistics
            if game_outcome in ["win", "loss", "draw"]:
                if 'outcomes' not in self.trainer.training_stats:
                    self.trainer.training_stats['outcomes'] = {'win': 0, 'loss': 0, 'draw': 0}
                self.trainer.training_stats['outcomes'][game_outcome] += 1

            # Add current model to the pool periodically
            if (episode + 1) % 100 == 0:
                # Save model for the pool
                pool_model_path = os.path.join(
                    self.trainer.model_dir,
                    f"model_pool_episode_{episode+1}.pt"
                )
                torch.save(self.trainer.policy_net.state_dict(), pool_model_path)

                # Create a copy for the model pool
                model_copy = DQN()
                model_copy.load_state_dict(self.trainer.policy_net.state_dict())

                # Create metadata for this model
                model_metadata = {
                    'episode': episode,
                    'reward': episode_reward,
                    'path': pool_model_path
                }

                # Add to pool or replace oldest model if pool is full
                if len(model_pool) < max_pool_size:
                    model_pool.append(model_copy)
                    model_pool_metadata.append(model_metadata)
                    print(f"Added model to pool (size: {len(model_pool)})")
                else:
                    # Find the oldest model to replace
                    oldest_idx = min(range(len(model_pool_metadata)),
                                    key=lambda i: model_pool_metadata[i]['episode'])

                    # Replace the model
                    model_pool[oldest_idx] = model_copy
                    model_pool_metadata[oldest_idx] = model_metadata
                    print(f"Replaced oldest model in pool (from episode {model_pool_metadata[oldest_idx]['episode']})")

                print(f"Model pool now contains {len(model_pool)} models from different training stages")

            # Evaluate against Stockfish levels periodically (every 1000 games)
            if (episode + 1) % eval_interval == 0 and self.trainer.stockfish_path and evaluation_stockfish:
                print(f"\n=== Evaluating model at episode {episode+1} against Stockfish levels {stockfish_levels} ===")

                # Save current model for evaluation
                eval_model_path = os.path.join(self.trainer.model_dir, f"model_checkpoint_episode_{episode+1}.pt")
                torch.save(self.trainer.policy_net.state_dict(), eval_model_path)
                print(f"Checkpoint saved to {eval_model_path}")

                # Run evaluation against all target levels
                results = self.trainer.evaluate_against_stockfish(
                    model_path=eval_model_path,
                    num_games=5,  # 5 games per level for faster evaluation
                    stockfish_levels=stockfish_levels
                )

                # Check performance against target levels
                if results:
                    print("\n=== Evaluation Results ===")
                    for level in sorted(results.keys()):
                        score = results[level]['score'] * 100
                        wins = results[level]['wins']
                        losses = results[level]['losses']
                        draws = results[level]['draws']
                        print(f"Stockfish Level {level}: Score {score:.1f}% | +{wins} -{losses} ={draws}")

                    # Check if we've reached target level
                    if target_level in results and results[target_level]['score'] >= 0.5:
                        print(f"\n!!! TARGET ACHIEVED: Model performs at Stockfish level {target_level} !!!")

                        # Save this model as a milestone
                        milestone_path = os.path.join(self.trainer.model_dir, f"model_milestone_level_{target_level}.pt")
                        torch.save(self.trainer.policy_net.state_dict(), milestone_path)
                        print(f"Milestone model saved to {milestone_path}")

                        # Optionally increase target level for next milestone
                        if target_level < max(stockfish_levels):
                            target_level += 1
                            print(f"New target level: {target_level}")

            # Print progress
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0

                # Calculate averages
                avg_reward = sum(self.trainer.training_stats['episode_rewards'][-10:]) / 10
                avg_length = sum(self.trainer.training_stats['episode_lengths'][-10:]) / 10
                avg_loss = sum(self.trainer.training_stats['losses'][-100:]) / max(len(self.trainer.training_stats['losses'][-100:]), 1) if self.trainer.training_stats['losses'] else 0

                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Loss: {avg_loss:.4f} | "
                      f"Positions: {total_positions} ({positions_per_second:.1f}/s) | "
                      f"Time: {elapsed_time/3600:.2f}h")

                # Update simple metrics
                self.trainer.metrics.update_reward(avg_reward)

                # Display metrics every 10 episodes
                if (episode + 1) % 10 == 0:
                    self.trainer.metrics.print_metrics()

            # Save model periodically
            if (episode + 1) % save_interval == 0:
                # Save regular checkpoint
                checkpoint_path = f"model_self_play_episode_{episode+1}.pt"
                self.trainer.save_model(checkpoint_path)
                print(f"Model checkpoint saved to {os.path.join(self.trainer.model_dir, checkpoint_path)}")

                # Save comprehensive checkpoint with optimizer state for resuming training
                comprehensive_checkpoint = {
                    'model_state_dict': self.trainer.policy_net.state_dict(),
                    'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                    'target_net_state_dict': self.trainer.target_net.state_dict(),
                    'training_stats': self.trainer.training_stats,
                    'episode': episode,
                    'total_positions': total_positions
                }
                comprehensive_path = os.path.join(self.trainer.model_dir, f"checkpoint_complete_{episode+1}.pt")
                torch.save(comprehensive_checkpoint, comprehensive_path)
                print(f"Comprehensive checkpoint saved to {comprehensive_path}")

                # Update best model if this is the best so far
                if episode_reward > max(self.trainer.training_stats['episode_rewards'][:-1], default=float('-inf')):
                    torch.save(self.trainer.policy_net.state_dict(), best_model_path)
                    print(f"New best model saved with reward: {episode_reward:.2f}")

                # Plot training metrics
                self.trainer.plot_metrics(output_dir=os.path.join(self.trainer.model_dir, 'metrics'))

                # Print current training metrics
                print("\n=== Current Training Metrics ===")
                self.trainer.print_training_metrics()

                # Calculate average loss for scheduler
                avg_loss = sum(self.trainer.training_stats['losses'][-100:]) / max(len(self.trainer.training_stats['losses'][-100:]), 1) if self.trainer.training_stats['losses'] else 0

                # Apply learning rate scheduling with loss value for ReduceLROnPlateau
                self.trainer.scheduler.step(avg_loss)

                # Get current learning rate
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")

                # Print detailed statistics
                if 'outcomes' in self.trainer.training_stats:
                    outcomes = self.trainer.training_stats['outcomes']
                    total_games = sum(outcomes.values())
                    if total_games > 0:
                        win_rate = outcomes['win'] / total_games * 100
                        draw_rate = outcomes['draw'] / total_games * 100
                        loss_rate = outcomes['loss'] / total_games * 100
                        print(f"Game outcomes: Win: {win_rate:.1f}% | Draw: {draw_rate:.1f}% | Loss: {loss_rate:.1f}%")

                # Evaluate the checkpoint model if Stockfish is available
                if self.trainer.stockfish_path:
                    print(f"\n=== Evaluating checkpoint model: {checkpoint_path} ===")
                    # Evaluate against target Stockfish levels
                    results = self.trainer.model_evaluator.evaluate_against_stockfish(
                        model_path=os.path.join(self.trainer.model_dir, checkpoint_path),
                        num_games=5,  # 5 games per level for more reliable evaluation
                        stockfish_levels=stockfish_levels  # Focus on target levels
                    )

                    # Print brief summary
                    if results:
                        # Determine approximate Stockfish level of the model
                        best_comparable_level = 0
                        for level in sorted(results.keys()):
                            if results[level]['score'] >= 0.45:  # Model is competitive (45%+ score)
                                best_comparable_level = level

                        strength_msg = f"Stockfish level {best_comparable_level}" if best_comparable_level > 0 else "below Stockfish level 1"
                        print(f"\nCheckpoint Evaluation: Model strength ~ {strength_msg}")

                        # Check if we've reached target level
                        if target_level in results and results[target_level]['score'] >= 0.5:
                            print(f"\n!!! TARGET ACHIEVED: Model performs at Stockfish level {target_level} (Score: {results[target_level]['score']*100:.1f}%) !!!")

        # Save final model
        self.trainer.save_model("model_self_play_final.pt")

        # Clean up
        if evaluation_stockfish:
            evaluation_stockfish.quit()

        print(f"Self-play training completed! Total positions: {total_positions}")
        print(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")

        # Generate and save metrics plots
        metrics_dir = os.path.join(self.trainer.model_dir, 'metrics')
        self.trainer.plot_metrics(output_dir=metrics_dir)

        return (self.trainer.training_stats['episode_rewards'],
                self.trainer.training_stats['episode_lengths'],
                self.trainer.training_stats['losses'])
