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
from utils import save_model, plot_training_progress

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
    Advanced self-play training with curriculum learning.
    """

    def __init__(self, trainer):
        """
        Initialize the self-play trainer.

        Args:
            trainer: ChessTrainer instance
        """
        self.trainer = trainer

    def train_self_play(self, num_episodes=5000, stockfish_opponent=True, stockfish_levels=None,
                      batch_size=64, save_interval=100, eval_interval=500, target_level=7):
        """
        Advanced self-play training with multiple opponents and adaptive difficulty.

        This optimized version includes:
        1. Training against Stockfish at various levels
        2. Self-play against previous best models
        3. Batch processing of experiences
        4. Adaptive difficulty based on performance
        5. Regular evaluation against target Stockfish level
        6. Advanced curriculum learning

        Args:
            num_episodes: Total number of episodes to train
            stockfish_opponent: Whether to use Stockfish as an opponent
            stockfish_levels: List of Stockfish levels to train against (default: [1,2,3,4,5,6,7])
            batch_size: Batch size for optimization
            save_interval: How often to save models (in episodes)
            eval_interval: How often to evaluate against target level (in episodes)
            target_level: Target Stockfish level to achieve

        Returns:
            Tuple of (episode_rewards, episode_lengths, losses)
        """
        print(f"Starting advanced self-play training for {num_episodes} episodes...")
        print(f"Target performance: Stockfish level {target_level}")

        # Set default Stockfish levels if not provided
        if stockfish_levels is None:
            stockfish_levels = list(range(1, target_level + 1))

        # Initialize tracking metrics
        self.trainer.training_stats['episode_rewards'] = []
        self.trainer.training_stats['episode_lengths'] = []
        self.trainer.training_stats['losses'] = []
        self.trainer.training_stats['win_rates'] = []

        # Initialize model pool for diverse opponents
        model_pool = []
        best_model_path = os.path.join(self.trainer.model_dir, "best_model.pt")

        # Track training progress
        start_time = time.time()
        total_positions = 0
        total_optimization_steps = 0

        # Enhanced adaptive curriculum learning parameters
        current_level = 1  # Start with easiest level
        current_sublevel = 0  # Sublevel for more granular progression (0-9)

        # Dynamic win rate thresholds based on level
        # Higher levels require higher win rates to advance
        base_win_rate_threshold = 0.55  # Base threshold for advancing
        win_rate_window = 30  # Calculate win rate over this many games (reduced for faster adaptation)

        # Track performance against each level and sublevel
        level_performance = {}  # Store performance metrics for each level
        wins_against_current_level = 0
        games_against_current_level = 0

        # Enhanced model pool parameters
        max_pool_size = 10  # Increased from 5 for more diversity
        model_pool_metadata = []  # Track metadata about each model in the pool

        # Create a separate Stockfish instance for opponents if needed
        opponent_stockfish = None
        if stockfish_opponent:
            try:
                # Use the stored Stockfish path
                opponent_stockfish = chess.engine.SimpleEngine.popen_uci(self.trainer.stockfish_path)
                # Configure Stockfish with base level and adjust time based on sublevel
                opponent_stockfish.configure({"Skill Level": current_level})

                # Adjust thinking time based on sublevel (0-9)
                # Lower sublevels get less thinking time, making them easier
                base_time = 0.05  # Base thinking time in seconds
                time_factor = 0.01 * current_sublevel  # 0.01s increments per sublevel
                thinking_time = base_time + time_factor

                print(f"Initialized opponent Stockfish at level {current_level}.{current_sublevel} using path: {self.trainer.stockfish_path}")
            except Exception as e:
                print(f"Error initializing opponent Stockfish: {e}")
                stockfish_opponent = False

        # Main training loop
        for episode in range(num_episodes):
            # Determine opponent type for this episode
            use_stockfish = False
            use_model_pool = False

            if stockfish_opponent:
                # 70% chance to play against Stockfish when available
                if random.random() < 0.7:
                    use_stockfish = True

                    # Occasionally play against higher levels for challenge (10% chance)
                    if random.random() < 0.1 and current_level < max(stockfish_levels):
                        # Challenge with higher level or higher sublevel
                        if current_level < max(stockfish_levels) - 1:
                            challenge_level = current_level + 1
                            challenge_sublevel = random.randint(0, 9)
                        else:
                            challenge_level = current_level
                            challenge_sublevel = min(current_sublevel + 5, 9)

                        # Configure challenge level
                        opponent_stockfish.configure({"Skill Level": challenge_level})

                        # Set challenge thinking time as a global variable for use in play method
                        global challenge_time
                        challenge_time = base_time + (0.01 * challenge_sublevel)

                        print(f"Challenge game against Stockfish level {challenge_level}.{challenge_sublevel}")
                    else:
                        # Regular game at current level and sublevel
                        opponent_stockfish.configure({"Skill Level": current_level})

            # Use model pool if available and not using Stockfish
            if not use_stockfish and len(model_pool) > 0 and random.random() < 0.8:
                use_model_pool = True
                # Select a random model from the pool
                opponent_model = random.choice(model_pool)
                opponent_model.eval()  # Set to evaluation mode

            # Initialize the game
            board = chess.Board()
            episode_reward = 0
            episode_length = 0
            positions_seen = {}  # Track positions for repetition detection
            experiences = []  # Collect experiences for batch processing

            # Play the game
            done = False
            while not done:
                # Get current state
                state = board_to_tensor(board, self.trainer.device)
                mask = create_move_mask(board, self.trainer.device)

                if board.turn == chess.WHITE:  # Our model plays as White
                    # Select action using our policy
                    action_idx, move = self.trainer.select_action(state, mask, board)
                else:  # Opponent plays as Black
                    if use_stockfish:
                        # Get move from Stockfish with adaptive thinking time
                        # Use challenge_time if available (for challenge games), otherwise use regular thinking time
                        if 'challenge_time' in globals():
                            thinking_time = challenge_time
                            # Reset after use to avoid affecting future regular games
                            del globals()['challenge_time']
                        else:
                            thinking_time = base_time + time_factor

                        result = opponent_stockfish.play(board, chess.engine.Limit(time=thinking_time))
                        move = result.move
                        action_idx = move.from_square * 64 + move.to_square
                    elif use_model_pool:
                        # Get move from opponent model
                        with torch.no_grad():
                            q_values = opponent_model(state, mask)
                            legal_moves = list(board.legal_moves)
                            legal_move_indices = [m.from_square * 64 + m.to_square for m in legal_moves]
                            legal_q_values = q_values[0, legal_move_indices]
                            best_idx = torch.argmax(legal_q_values).item()
                            move = legal_moves[best_idx]
                            action_idx = legal_move_indices[best_idx]
                    else:
                        # Self-play against current policy
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

                # Calculate reward using adaptive sampling
                # The reward calculator will determine position importance internally
                # and adjust evaluation depth accordingly
                reward = self.trainer.reward_calculator.calculate_stockfish_reward(board, prev_board)

                # Handle repetitions
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

            # Update win statistics for curriculum learning
            if use_stockfish:
                games_against_current_level += 1
                if board.is_checkmate() and board.turn == chess.BLACK:  # Model (White) won
                    wins_against_current_level += 1

                # Calculate win rate over the window
                if games_against_current_level >= win_rate_window:
                    win_rate = wins_against_current_level / games_against_current_level
                    self.trainer.training_stats['win_rates'].append(win_rate)

                    # Calculate dynamic win rate threshold based on level
                    # Higher levels require higher win rates to advance
                    win_rate_threshold = base_win_rate_threshold + (current_level * 0.02)

                    # Store performance metrics for this level
                    level_key = f"{current_level}.{current_sublevel}"
                    if level_key not in level_performance:
                        level_performance[level_key] = []
                    level_performance[level_key].append(win_rate)

                    # Advance to next sublevel or level if win rate is high enough
                    if win_rate >= win_rate_threshold:
                        # Save model before advancing
                        sublevel_model_path = os.path.join(
                            self.trainer.model_dir,
                            f"model_level_{current_level}_{current_sublevel}.pt"
                        )
                        torch.save(self.trainer.policy_net.state_dict(), sublevel_model_path)

                        # Add current model to the pool with metadata
                        model_copy = DQN()
                        model_copy.load_state_dict(self.trainer.policy_net.state_dict())

                        # Create metadata for this model
                        model_metadata = {
                            'level': current_level,
                            'sublevel': current_sublevel,
                            'win_rate': win_rate,
                            'episode': episode,
                            'path': sublevel_model_path
                        }

                        # Add to pool or replace least successful model if pool is full
                        if len(model_pool) < max_pool_size:
                            model_pool.append(model_copy)
                            model_pool_metadata.append(model_metadata)
                            print(f"Added model to pool (size: {len(model_pool)})")
                        else:
                            # Find the least successful model to replace
                            # Use a combination of level and win rate to determine
                            replacement_scores = [
                                (i, meta['level'] + meta['win_rate'])
                                for i, meta in enumerate(model_pool_metadata)
                            ]
                            replacement_scores.sort(key=lambda x: x[1])  # Sort by score
                            idx = replacement_scores[0][0]  # Get index of lowest scoring model

                            # Replace the model
                            model_pool[idx] = model_copy
                            model_pool_metadata[idx] = model_metadata
                            print(f"Replaced model in pool at position {idx}")

                        # Advance sublevel or level
                        if current_sublevel < 9:
                            # Advance sublevel
                            current_sublevel += 1
                            print(f"\n=== ADVANCING TO SUBLEVEL {current_level}.{current_sublevel} ===")
                            print(f"Win rate at level {current_level}.{current_sublevel-1}: {win_rate*100:.1f}%")
                        elif current_level < max(stockfish_levels):
                            # Advance to next level
                            current_level += 1
                            current_sublevel = 0
                            print(f"\n=== ADVANCING TO STOCKFISH LEVEL {current_level}.{current_sublevel} ===")
                            print(f"Win rate at level {current_level-1}.9: {win_rate*100:.1f}%")

                        # Reset statistics for new level/sublevel
                        wins_against_current_level = 0
                        games_against_current_level = 0

            # Evaluate against target level periodically
            if stockfish_opponent and (episode + 1) % eval_interval == 0:
                print(f"\n=== Evaluating against Stockfish level {target_level} ===")
                # Save current model for evaluation
                eval_model_path = os.path.join(self.trainer.model_dir, "temp_eval_model.pt")
                torch.save(self.trainer.policy_net.state_dict(), eval_model_path)

                # Run evaluation
                results = self.trainer.evaluate_against_stockfish(
                    model_path=eval_model_path,
                    num_games=5,
                    stockfish_levels=[target_level]
                )

                # Check if we've reached target level
                if results and target_level in results:
                    score = results[target_level]['score']
                    if score >= 0.5:  # 50% or better score against target
                        print(f"\n!!! TARGET ACHIEVED: Model performs at Stockfish level {target_level} !!!")
                        print(f"Score: {score*100:.1f}%")

                        # Save this model as a milestone
                        milestone_path = os.path.join(self.trainer.model_dir, f"model_level_{target_level}.pt")
                        torch.save(self.trainer.policy_net.state_dict(), milestone_path)
                        print(f"Milestone model saved to {milestone_path}")

                        # Optionally increase target level
                        if target_level < 10:
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

                print(f"Episode {episode+1}/{num_episodes} | Level: {current_level}/{max(stockfish_levels)} | "
                      f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Loss: {avg_loss:.4f} | "
                      f"Positions: {total_positions} ({positions_per_second:.1f}/s) | "
                      f"Time: {elapsed_time/3600:.2f}h")

            # Save model periodically
            if (episode + 1) % save_interval == 0:
                checkpoint_path = f"model_self_play_episode_{episode+1}.pt"
                self.trainer.save_model(checkpoint_path)

                # Update best model if this is the best so far
                if episode_reward > max(self.trainer.training_stats['episode_rewards'][:-1], default=float('-inf')):
                    torch.save(self.trainer.policy_net.state_dict(), best_model_path)
                    print(f"New best model saved with reward: {episode_reward:.2f}")

                # Plot training progress
                plot_training_progress(self.trainer.training_stats,
                                      os.path.join(self.trainer.model_dir, 'training_progress.png'))

                # Calculate average loss for scheduler
                avg_loss = sum(self.trainer.training_stats['losses'][-100:]) / max(len(self.trainer.training_stats['losses'][-100:]), 1) if self.trainer.training_stats['losses'] else 0

                # Apply learning rate scheduling with loss value for ReduceLROnPlateau
                self.trainer.scheduler.step(avg_loss)

                # Get current learning rate
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")

                # Evaluate the checkpoint model if Stockfish is available
                if self.trainer.stockfish_path:
                    print(f"\n=== Evaluating checkpoint model: {checkpoint_path} ===")
                    # Use a smaller number of games for faster evaluation during training
                    results = self.trainer.model_evaluator.evaluate_against_stockfish(
                        model_path=os.path.join(self.trainer.model_dir, checkpoint_path),
                        num_games=3,  # Fewer games for quicker evaluation
                        stockfish_levels=[1, 3, 5, target_level]  # Test against key levels including target
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
        if opponent_stockfish:
            opponent_stockfish.quit()

        print(f"Advanced self-play training completed! Total positions: {total_positions}")
        print(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")

        return (self.trainer.training_stats['episode_rewards'],
                self.trainer.training_stats['episode_lengths'],
                self.trainer.training_stats['losses'])
