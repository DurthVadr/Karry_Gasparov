"""
Comprehensive chess model evaluation script.

This script provides a command-line interface for evaluating chess models
against Stockfish at different skill levels and generating detailed reports.
"""

import os
import argparse
import time
import torch
import chess
import chess.engine
import chess.pgn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drl_agent import ChessAgent
from evaluation import ModelEvaluator
from visualization import plot_evaluation_results
from model_integration import ModelIntegration

def evaluate_against_stockfish(model_path, stockfish_path, num_games=10, min_level=1, max_level=10, 
                              output_dir="evaluation_results", save_plots=True):
    """
    Evaluate a chess model against Stockfish at different skill levels.
    
    Args:
        model_path (str): Path to the model file to evaluate
        stockfish_path (str): Path to the Stockfish executable
        num_games (int): Number of games to play against each Stockfish level
        min_level (int): Minimum Stockfish level to test against
        max_level (int): Maximum Stockfish level to test against
        output_dir (str): Directory to save evaluation results
        save_plots (bool): Whether to save evaluation plots
        
    Returns:
        dict: Dictionary with results for each Stockfish level
    """
    print(f"\n=== Evaluating Model Against Stockfish: {os.path.basename(model_path)} ===")
    print(f"Playing {num_games} games against Stockfish levels {min_level}-{max_level}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize model evaluator
    evaluator = ModelEvaluator(stockfish_path)
    
    # Start evaluation
    start_time = time.time()
    results = evaluator.evaluate_against_stockfish(
        model_path=model_path,
        num_games=num_games,
        stockfish_levels=range(min_level, max_level + 1)
    )
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    # Save results to file
    if results:
        # Generate timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path).replace('.pt', '').replace('.pth', '')
        
        # Save summary to text file
        summary_path = os.path.join(output_dir, f"{model_name}_stockfish_eval_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Evaluation of model: {model_path}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Games per level: {num_games}\n")
            f.write(f"Evaluation time: {eval_time:.1f} seconds\n\n")
            
            f.write("=== Results Summary ===\n")
            for level in sorted(results.keys()):
                win_rate = results[level]['win_rate'] * 100
                draw_rate = results[level]['draws'] / num_games * 100
                loss_rate = results[level]['losses'] / num_games * 100
                score = results[level]['score'] * 100
                
                f.write(f"Stockfish Level {level}:\n")
                f.write(f"  Wins: {results[level]['wins']}/{num_games} ({win_rate:.1f}%)\n")
                f.write(f"  Draws: {results[level]['draws']}/{num_games} ({draw_rate:.1f}%)\n")
                f.write(f"  Losses: {results[level]['losses']}/{num_games} ({loss_rate:.1f}%)\n")
                f.write(f"  Score: {score:.1f}%\n\n")
            
            # Determine approximate Stockfish level of the model
            best_comparable_level = 0
            for level in sorted(results.keys()):
                if results[level]['score'] >= 0.45:  # Model is competitive (45%+ score)
                    best_comparable_level = level
            
            f.write("=== Model Strength Assessment ===\n")
            if best_comparable_level > 0:
                f.write(f"The model plays approximately at Stockfish level {best_comparable_level} strength\n")
                if results[best_comparable_level]['score'] > 0.55:
                    f.write(f"The model is better than Stockfish level {best_comparable_level}\n")
                elif results[best_comparable_level]['score'] >= 0.45:
                    f.write(f"The model is almost as good as Stockfish level {best_comparable_level}\n")
                else:
                    f.write(f"The model is not quite at Stockfish level {best_comparable_level} yet\n")
            else:
                f.write("The model is below Stockfish level 1 strength\n")
        
        print(f"Evaluation summary saved to {summary_path}")
        
        # Save plot if requested
        if save_plots:
            plot_path = os.path.join(output_dir, f"{model_name}_stockfish_eval_plot_{timestamp}.png")
            plot_evaluation_results(results, plot_path)
            print(f"Evaluation plot saved to {plot_path}")
    
    # Clean up
    evaluator.close()
    
    return results

def evaluate_on_positions(model_path, num_positions=5, num_moves_per_position=5, output_dir="evaluation_results"):
    """
    Evaluate a model on a set of random positions.
    
    Args:
        model_path (str): Path to the model file to evaluate
        num_positions (int): Number of random positions to evaluate
        num_moves_per_position (int): Number of moves to play from each position
        output_dir (str): Directory to save evaluation results
    """
    print(f"\n=== Evaluating Model On Positions: {os.path.basename(model_path)} ===")
    print(f"Testing on {num_positions} random positions, playing {num_moves_per_position} moves from each")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    model_integration = ModelIntegration()
    try:
        model = model_integration.load_model(model_path)
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return
    
    # Generate random positions by playing random moves
    positions = []
    for i in range(num_positions):
        board = chess.Board()
        # Play 5-15 random moves to get to a middle-game position
        num_random_moves = np.random.randint(5, 15)
        for _ in range(num_random_moves):
            if board.is_game_over():
                break
            move = np.random.choice(list(board.legal_moves))
            board.push(move)
        positions.append(board.copy())
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path).replace('.pt', '').replace('.pth', '')
    
    # Save results to file
    report_path = os.path.join(output_dir, f"{model_name}_position_eval_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Position Evaluation of model: {model_path}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of positions: {num_positions}\n")
        f.write(f"Moves per position: {num_moves_per_position}\n\n")
        
        # Evaluate each position
        for i, board in enumerate(positions):
            f.write(f"{'='*50}\n")
            f.write(f"Position {i+1}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Starting FEN: {board.fen()}\n\n")
            f.write(f"Starting position:\n{board}\n\n")
            
            # Create a copy of the board for evaluation
            board_copy = board.copy()
            
            # Play moves from this position
            for j in range(num_moves_per_position):
                if board_copy.is_game_over():
                    f.write(f"Game over: {board_copy.result()}\n")
                    break
                
                f.write(f"Move {j+1}: {'White' if board_copy.turn == chess.WHITE else 'Black'} to move\n")
                
                # Get move from model
                move = model.get_move(board_copy)
                f.write(f"Model plays: {move.uci()} ({board_copy.san(move)})\n")
                
                # Make the move
                board_copy.push(move)
                f.write(f"Board after move:\n{board_copy}\n\n")
            
            f.write("\n\n")
    
    print(f"Position evaluation saved to {report_path}")

def compare_models(model_paths, num_positions=3, num_moves_per_position=5, output_dir="evaluation_results"):
    """
    Compare multiple models on the same positions.
    
    Args:
        model_paths (list): List of paths to model files to compare
        num_positions (int): Number of random positions to evaluate
        num_moves_per_position (int): Number of moves to play from each position
        output_dir (str): Directory to save evaluation results
    """
    print(f"\n=== Comparing Models ===")
    print(f"Testing {len(model_paths)} models on {num_positions} random positions")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create model integration
    model_integration = ModelIntegration()
    
    # Load models
    models = []
    for path in model_paths:
        try:
            model = model_integration.load_model(path)
            print(f"Loaded model: {path}")
            models.append((path, model))
        except Exception as e:
            print(f"Error loading model {path}: {str(e)}")
    
    if not models:
        print("No models could be loaded. Aborting comparison.")
        return
    
    # Generate random positions by playing random moves
    positions = []
    for i in range(num_positions):
        board = chess.Board()
        # Play 5-15 random moves to get to a middle-game position
        num_random_moves = np.random.randint(5, 15)
        for _ in range(num_random_moves):
            if board.is_game_over():
                break
            move = np.random.choice(list(board.legal_moves))
            board.push(move)
        positions.append(board.copy())
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save results to file
    report_path = os.path.join(output_dir, f"model_comparison_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model Comparison\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models compared: {', '.join([os.path.basename(path) for path, _ in models])}\n")
        f.write(f"Number of positions: {num_positions}\n")
        f.write(f"Moves per position: {num_moves_per_position}\n\n")
        
        # Evaluate each model on each position
        for i, board in enumerate(positions):
            f.write(f"{'='*50}\n")
            f.write(f"Position {i+1}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Starting FEN: {board.fen()}\n\n")
            f.write(f"Starting position:\n{board}\n\n")
            
            for path, model in models:
                f.write(f"{'-'*30}\n")
                f.write(f"Model: {os.path.basename(path)}\n")
                f.write(f"{'-'*30}\n")
                
                # Create a copy of the board for this model
                board_copy = board.copy()
                
                # Play moves from this position
                for j in range(num_moves_per_position):
                    if board_copy.is_game_over():
                        f.write(f"Game over: {board_copy.result()}\n")
                        break
                    
                    f.write(f"Move {j+1}: {'White' if board_copy.turn == chess.WHITE else 'Black'} to move\n")
                    
                    # Get move from model
                    move = model.get_move(board_copy)
                    f.write(f"Model plays: {move.uci()} ({board_copy.san(move)})\n")
                    
                    # Make the move
                    board_copy.push(move)
                    f.write(f"Board after move:\n{board_copy}\n\n")
                
                f.write("\n")
            
            f.write("\n\n")
    
    print(f"Model comparison saved to {report_path}")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Comprehensive chess model evaluation")
    
    # Common arguments
    parser.add_argument("--model_path", required=True, help="Path to the model file to evaluate")
    parser.add_argument("--output_dir", default="evaluation_results", help="Directory to save evaluation results")
    
    # Evaluation mode
    parser.add_argument("--mode", choices=["stockfish", "positions", "compare"], default="stockfish",
                       help="Evaluation mode: stockfish (vs Stockfish), positions (on random positions), compare (with other models)")
    
    # Stockfish evaluation arguments
    parser.add_argument("--stockfish_path", help="Path to Stockfish executable (required for stockfish mode)")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games to play against each Stockfish level")
    parser.add_argument("--min_level", type=int, default=1, help="Minimum Stockfish level to test against")
    parser.add_argument("--max_level", type=int, default=10, help="Maximum Stockfish level to test against")
    
    # Position evaluation arguments
    parser.add_argument("--num_positions", type=int, default=5, help="Number of random positions to evaluate")
    parser.add_argument("--num_moves", type=int, default=5, help="Number of moves to play from each position")
    
    # Model comparison arguments
    parser.add_argument("--compare_with", nargs="+", help="Paths to other model files to compare with")
    
    # Other options
    parser.add_argument("--no_plots", action="store_true", help="Don't save evaluation plots")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Run evaluation based on mode
    if args.mode == "stockfish":
        # Validate Stockfish path
        if not args.stockfish_path or not os.path.exists(args.stockfish_path):
            print(f"Error: Stockfish executable not found: {args.stockfish_path}")
            print("Please provide a valid Stockfish path with --stockfish_path")
            return
        
        evaluate_against_stockfish(
            model_path=args.model_path,
            stockfish_path=args.stockfish_path,
            num_games=args.num_games,
            min_level=args.min_level,
            max_level=args.max_level,
            output_dir=args.output_dir,
            save_plots=not args.no_plots
        )
    
    elif args.mode == "positions":
        evaluate_on_positions(
            model_path=args.model_path,
            num_positions=args.num_positions,
            num_moves_per_position=args.num_moves,
            output_dir=args.output_dir
        )
    
    elif args.mode == "compare":
        if not args.compare_with:
            print("Error: No models to compare with. Please provide paths with --compare_with")
            return
        
        # Add the main model to the list
        model_paths = [args.model_path] + args.compare_with
        
        compare_models(
            model_paths=model_paths,
            num_positions=args.num_positions,
            num_moves_per_position=args.num_moves,
            output_dir=args.output_dir
        )
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
