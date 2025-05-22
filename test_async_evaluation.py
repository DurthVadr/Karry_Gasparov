"""
Test script for async evaluation and caching improvements.

This script tests the performance of the async evaluation system with the new
caching mechanism and increased worker count.
"""

import os
import time
import chess
import argparse
import random
from async_evaluation import AsyncStockfishEvaluator

def generate_random_position(num_moves=20):
    """Generate a random chess position by playing random moves."""
    board = chess.Board()
    
    # Play random moves
    for _ in range(min(num_moves, 40)):  # Limit to 40 moves to avoid very long games
        if board.is_game_over():
            break
        
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            board.push(move)
    
    return board

def test_cache_performance(stockfish_path, num_positions=100, num_repeats=3, num_workers=8, cache_size=10000):
    """
    Test the performance of the async evaluation cache.
    
    Args:
        stockfish_path: Path to Stockfish executable
        num_positions: Number of random positions to generate
        num_repeats: Number of times to repeat the evaluation of each position
        num_workers: Number of worker threads to use
        cache_size: Size of the evaluation cache
    """
    print(f"\n=== Testing Async Evaluation Cache Performance ===")
    print(f"Stockfish path: {stockfish_path}")
    print(f"Number of positions: {num_positions}")
    print(f"Number of repeats: {num_repeats}")
    print(f"Number of workers: {num_workers}")
    print(f"Cache size: {cache_size}")
    
    # Initialize the async evaluator
    evaluator = AsyncStockfishEvaluator(
        stockfish_path=stockfish_path,
        num_workers=num_workers,
        cache_size=cache_size
    )
    
    if not evaluator.running:
        print("Error: Async evaluator failed to initialize")
        return
    
    # Generate random positions
    print("\nGenerating random positions...")
    positions = [generate_random_position() for _ in range(num_positions)]
    print(f"Generated {len(positions)} random positions")
    
    # First evaluation (should be all cache misses)
    print("\nFirst evaluation (no cache hits expected)...")
    start_time = time.time()
    results1 = evaluator.evaluate_positions_batch(positions, depth=10, use_cache=True)
    first_eval_time = time.time() - start_time
    
    print(f"First evaluation completed in {first_eval_time:.2f} seconds")
    evaluator.print_cache_stats()
    
    # Repeat evaluations (should be mostly cache hits)
    for i in range(num_repeats):
        print(f"\nRepeat evaluation #{i+1} (high cache hits expected)...")
        start_time = time.time()
        results_repeat = evaluator.evaluate_positions_batch(positions, depth=10, use_cache=True)
        repeat_eval_time = time.time() - start_time
        
        print(f"Repeat evaluation #{i+1} completed in {repeat_eval_time:.2f} seconds")
        print(f"Speed improvement: {first_eval_time / max(repeat_eval_time, 0.001):.2f}x faster")
        evaluator.print_cache_stats()
        
        # Verify results are consistent
        if results1 == results_repeat:
            print("Results are consistent with first evaluation ✓")
        else:
            print("WARNING: Results differ from first evaluation!")
    
    # Test with cache disabled
    print("\nEvaluation with cache disabled...")
    start_time = time.time()
    results_no_cache = evaluator.evaluate_positions_batch(positions, depth=10, use_cache=False)
    no_cache_time = time.time() - start_time
    
    print(f"No-cache evaluation completed in {no_cache_time:.2f} seconds")
    print(f"Cache speedup: {no_cache_time / max(repeat_eval_time, 0.001):.2f}x faster with cache")
    
    # Clean up
    evaluator.close()
    
    print("\n=== Test completed ===")

def test_worker_scaling(stockfish_path, num_positions=50, worker_counts=[1, 2, 4, 8]):
    """
    Test how performance scales with different numbers of worker threads.
    
    Args:
        stockfish_path: Path to Stockfish executable
        num_positions: Number of random positions to generate
        worker_counts: List of worker thread counts to test
    """
    print(f"\n=== Testing Worker Thread Scaling ===")
    print(f"Stockfish path: {stockfish_path}")
    print(f"Number of positions: {num_positions}")
    print(f"Worker counts to test: {worker_counts}")
    
    # Generate random positions
    print("\nGenerating random positions...")
    positions = [generate_random_position() for _ in range(num_positions)]
    print(f"Generated {len(positions)} random positions")
    
    results = {}
    
    # Test each worker count
    for num_workers in worker_counts:
        print(f"\nTesting with {num_workers} worker threads...")
        
        # Initialize the async evaluator with this worker count
        evaluator = AsyncStockfishEvaluator(
            stockfish_path=stockfish_path,
            num_workers=num_workers,
            cache_size=10000
        )
        
        if not evaluator.running:
            print(f"Error: Async evaluator failed to initialize with {num_workers} workers")
            continue
        
        # Evaluate positions
        start_time = time.time()
        evaluator.evaluate_positions_batch(positions, depth=10, use_cache=False)
        eval_time = time.time() - start_time
        
        # Store results
        results[num_workers] = eval_time
        
        print(f"Evaluation with {num_workers} workers completed in {eval_time:.2f} seconds")
        print(f"Positions per second: {num_positions / eval_time:.2f}")
        
        # Clean up
        evaluator.close()
    
    # Print summary
    print("\n=== Worker Scaling Summary ===")
    baseline = results.get(worker_counts[0], 0)
    for num_workers, eval_time in sorted(results.items()):
        speedup = baseline / max(eval_time, 0.001) if baseline > 0 else 0
        print(f"{num_workers} workers: {eval_time:.2f} seconds, {speedup:.2f}x speedup")
    
    print("\n=== Test completed ===")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Test async evaluation and caching")
    
    # Common arguments
    parser.add_argument("--stockfish_path", required=True, help="Path to Stockfish executable")
    
    # Test mode
    parser.add_argument("--mode", choices=["cache", "workers", "both"], default="both",
                       help="Test mode: cache (test caching), workers (test worker scaling), both (run both tests)")
    
    # Cache test parameters
    parser.add_argument("--num_positions", type=int, default=50, help="Number of positions to test")
    parser.add_argument("--num_repeats", type=int, default=3, help="Number of times to repeat evaluation")
    parser.add_argument("--cache_size", type=int, default=10000, help="Size of evaluation cache")
    
    # Worker test parameters
    parser.add_argument("--worker_counts", type=int, nargs="+", default=[1, 2, 4, 8], 
                       help="Worker counts to test")
    
    args = parser.parse_args()
    
    # Validate Stockfish path
    if not os.path.exists(args.stockfish_path):
        print(f"Error: Stockfish executable not found: {args.stockfish_path}")
        return
    
    # Run tests based on mode
    if args.mode in ["cache", "both"]:
        test_cache_performance(
            stockfish_path=args.stockfish_path,
            num_positions=args.num_positions,
            num_repeats=args.num_repeats,
            cache_size=args.cache_size
        )
    
    if args.mode in ["workers", "both"]:
        test_worker_scaling(
            stockfish_path=args.stockfish_path,
            num_positions=args.num_positions,
            worker_counts=args.worker_counts
        )

if __name__ == "__main__":
    main()
