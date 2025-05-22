"""
Simple asynchronous evaluation module for chess reinforcement learning.

This module provides basic asynchronous Stockfish evaluation using worker threads.
"""

import os
import time
import queue
import threading
import chess
import chess.engine
from collections import OrderedDict

class AsyncStockfishEvaluator:
    """
    Asynchronous Stockfish evaluator using worker threads.

    This class creates a pool of Stockfish engines running in separate threads
    to evaluate positions in parallel.
    """

    def __init__(self, stockfish_path=None, num_workers=4, cache_size=10000):
        """
        Initialize the asynchronous Stockfish evaluator.

        Args:
            stockfish_path (str, optional): Path to Stockfish executable
            num_workers (int, optional): Number of worker threads to create
            cache_size (int, optional): Maximum number of positions to cache
        """
        self.stockfish_path = stockfish_path
        self.num_workers = num_workers
        self.workers = []
        self.engines = []
        self.running = False

        # Queues for communication between threads
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Simple cache for storing evaluation results
        self.cache_size = cache_size
        self.eval_cache = OrderedDict()
        self.cache_lock = threading.Lock()  # Lock for thread-safe cache access

        # Basic statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'completed_evals': 0
        }

        # Initialize workers if Stockfish path is provided
        if stockfish_path and os.path.exists(stockfish_path):
            self.initialize_workers()
        else:
            print("Stockfish path not provided or invalid. Asynchronous evaluation not available.")

    def initialize_workers(self):
        """Initialize worker threads and Stockfish engines."""
        try:
            # Create worker threads
            self.running = True
            for i in range(self.num_workers):
                worker_thread = threading.Thread(
                    target=self._worker_function,
                    args=(i,),
                    daemon=True
                )
                self.workers.append(worker_thread)
                worker_thread.start()

            print(f"Initialized {self.num_workers} asynchronous Stockfish worker threads")
        except Exception as e:
            print(f"Error initializing worker threads: {e}")
            self.running = False

    def _worker_function(self, worker_id):
        """
        Worker thread function that processes evaluation requests.

        Args:
            worker_id (int): ID of the worker thread
        """
        try:
            # Initialize Stockfish engine for this worker
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engines.append(engine)

            print(f"Worker {worker_id}: Stockfish engine initialized")

            # Process requests until stopped
            while self.running:
                try:
                    # Get a request from the queue with timeout
                    request = self.request_queue.get(timeout=0.1)

                    # Check for termination signal
                    if request is None:
                        break

                    # Unpack the request
                    request_id, board, depth, time_limit, position_key = request

                    # Evaluate the position
                    start_time = time.time()
                    result = engine.analyse(board=board, limit=chess.engine.Limit(depth=depth))

                    # Extract the score
                    score = result['score'].relative.score(mate_score=10000)

                    # Update the cache with the result
                    with self.cache_lock:
                        # Add to cache
                        self.eval_cache[position_key] = score

                        # If cache is too large, remove oldest entries (LRU policy)
                        if len(self.eval_cache) > self.cache_size:
                            # Remove the first item (oldest) from the OrderedDict
                            self.eval_cache.popitem(last=False)

                    # Put the result in the result queue
                    self.result_queue.put((request_id, score, time.time() - start_time))

                    # Mark the request as done
                    self.request_queue.task_done()

                except queue.Empty:
                    # No requests in the queue, continue waiting
                    continue
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
                    # Put an error result in the queue
                    self.result_queue.put((request_id, None, 0))
                    self.request_queue.task_done()

        except Exception as e:
            print(f"Worker {worker_id} failed to initialize: {e}")

        finally:
            # Clean up the engine when the thread exits
            if worker_id < len(self.engines) and self.engines[worker_id]:
                try:
                    self.engines[worker_id].quit()
                except:
                    pass

    def evaluate_position(self, board, depth=8, time_limit=None, use_cache=True):
        """
        Evaluate a chess position asynchronously.

        Args:
            board (chess.Board): The chess position to evaluate
            depth (int, optional): Depth for Stockfish analysis
            time_limit (float, optional): Time limit for analysis in seconds (not used)
            use_cache (bool, optional): Whether to use cached results

        Returns:
            int: Request ID that can be used to retrieve the result later
        """
        if not self.running:
            raise RuntimeError("Asynchronous evaluator is not running")

        # Generate a unique key for the position
        position_key = board.fen()

        # Check cache if enabled
        if use_cache:
            with self.cache_lock:
                if position_key in self.eval_cache:
                    # Move this item to the end of the OrderedDict (most recently used)
                    value = self.eval_cache.pop(position_key)
                    self.eval_cache[position_key] = value
                    self.stats['cache_hits'] += 1
                    return value
                else:
                    self.stats['cache_misses'] += 1

        # Generate a unique request ID
        request_id = self.stats['total_requests']
        self.stats['total_requests'] += 1

        # Put the request in the queue (time_limit is kept for compatibility)
        self.request_queue.put((request_id, board.copy(), depth, time_limit, position_key))

        return request_id

    def get_result(self, request_id=None, block=True, timeout=None):
        """
        Get the result of an evaluation request.

        Args:
            request_id (int, optional): ID of the request to get the result for.
                                       If None, get the next available result.
            block (bool, optional): Whether to block until a result is available
            timeout (float, optional): Maximum time to wait for a result

        Returns:
            tuple: (request_id, score, evaluation_time) or None if no result is available
        """
        if not self.running:
            raise RuntimeError("Asynchronous evaluator is not running")

        try:
            # Get a result from the queue
            result = self.result_queue.get(block=block, timeout=timeout)

            # If we're looking for a specific request and this isn't it,
            # put the result back in the queue and try again
            if request_id is not None and result[0] != request_id:
                self.result_queue.put(result)
                return None

            # Update statistics
            self.stats['completed_evals'] += 1

            # Mark the result as processed
            self.result_queue.task_done()

            return result

        except queue.Empty:
            return None

    def evaluate_positions_batch(self, boards, depth=8, time_limit=None, use_cache=True):
        """
        Evaluate a batch of positions and wait for all results.

        Args:
            boards (list): List of chess.Board objects to evaluate
            depth (int, optional): Depth for Stockfish analysis
            time_limit (float, optional): Time limit for analysis in seconds (not used)
            use_cache (bool, optional): Whether to use cached results

        Returns:
            list: List of evaluation scores in the same order as the input boards
        """
        if not self.running:
            raise RuntimeError("Asynchronous evaluator is not running")

        # Simple implementation - evaluate each position and collect results
        results = []

        for board in boards:
            # Submit for evaluation (will use cache if enabled)
            request_id = self.evaluate_position(board, depth, time_limit, use_cache)

            # If we got a direct result from cache
            if isinstance(request_id, (int, float)):
                results.append(request_id)
            else:
                # Wait for the result
                result = self.get_result(request_id, block=True)
                if result:
                    results.append(result[1])  # Extract score
                else:
                    results.append(0)  # Default value if evaluation failed

        return results

    def print_stats(self):
        """Print cache statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / max(total_requests, 1) * 100

        print("\n=== Evaluation Cache Statistics ===")
        print(f"Cache size: {len(self.eval_cache)} / {self.cache_size} positions")
        print(f"Cache hits: {self.stats['cache_hits']} ({hit_rate:.1f}%)")
        print(f"Cache misses: {self.stats['cache_misses']}")
        print(f"Total requests: {total_requests}")
        print(f"Positions evaluated: {self.stats['completed_evals']}")
        print("===================================\n")

    def close(self):
        """Clean up resources and stop worker threads."""
        if self.running:
            self.running = False

            # Send termination signal to all workers
            for _ in range(self.num_workers):
                self.request_queue.put(None)

            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=1.0)

            # Clean up engines
            for engine in self.engines:
                try:
                    engine.quit()
                except:
                    pass

            self.workers = []
            self.engines = []

            print("Asynchronous Stockfish evaluator shut down")

    def get_stats(self):
        """Get statistics about the evaluator's performance."""
        return self.stats
