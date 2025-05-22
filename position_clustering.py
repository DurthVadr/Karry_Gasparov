"""
Position clustering module for improved cache hit rates.

This module provides functions for clustering similar chess positions
to improve cache hit rates in the evaluation cache.
"""

import chess
import numpy as np
from collections import defaultdict, OrderedDict

class PositionClusterCache:
    """
    Cache with position clustering for improved hit rates.
    
    This class implements a cache that clusters similar chess positions
    to improve cache hit rates. It uses a similarity metric to determine
    if a position is similar enough to a cached position to use the cached
    evaluation.
    """
    
    def __init__(self, cache_size=20000, similarity_threshold=0.85):
        """
        Initialize the position cluster cache.
        
        Args:
            cache_size (int): Maximum number of positions to cache
            similarity_threshold (float): Threshold for position similarity (0-1)
        """
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        
        # Main cache (OrderedDict for LRU behavior)
        self.cache = OrderedDict()
        
        # Cluster mapping (position hash -> cluster ID)
        self.position_clusters = {}
        
        # Cluster data (cluster ID -> list of positions)
        self.clusters = defaultdict(list)
        
        # Cluster count
        self.cluster_count = 0
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'cluster_hits': 0,
            'total_lookups': 0,
            'clusters_created': 0,
            'positions_added': 0
        }
    
    def get(self, board):
        """
        Get a value from the cache, considering position clustering.
        
        Args:
            board (chess.Board): The board position to look up
            
        Returns:
            tuple: (value, hit_type) where hit_type is 'exact', 'cluster', or None
        """
        self.stats['total_lookups'] += 1
        
        # Try exact match first
        position_key = board.fen()
        if position_key in self.cache:
            # Move to end of OrderedDict (most recently used)
            value = self.cache.pop(position_key)
            self.cache[position_key] = value
            self.stats['hits'] += 1
            return value, 'exact'
        
        # If no exact match, try cluster match
        position_hash = self._get_position_hash(board)
        
        # Check if this position is in a known cluster
        if position_hash in self.position_clusters:
            cluster_id = self.position_clusters[position_hash]
            
            # Find the most similar position in the cluster
            best_match = None
            best_similarity = 0
            
            for cluster_pos, cluster_value in self.clusters[cluster_id]:
                similarity = self._calculate_similarity(board, cluster_pos)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cluster_value
            
            # If we found a similar enough position, use its value
            if best_match is not None and best_similarity >= self.similarity_threshold:
                self.stats['cluster_hits'] += 1
                return best_match, 'cluster'
        
        # No match found
        self.stats['misses'] += 1
        return None, None
    
    def put(self, board, value):
        """
        Add a position to the cache with clustering.
        
        Args:
            board (chess.Board): The board position
            value: The value to cache
        """
        # Add to main cache
        position_key = board.fen()
        self.cache[position_key] = value
        
        # Add to position clusters
        position_hash = self._get_position_hash(board)
        
        # Find the most similar existing cluster
        best_cluster = None
        best_similarity = 0
        
        for cluster_id, positions in self.clusters.items():
            if not positions:
                continue
                
            # Check similarity with the first position in the cluster
            similarity = self._calculate_similarity(board, positions[0][0])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        
        # If we found a similar enough cluster, add to it
        if best_cluster is not None and best_similarity >= self.similarity_threshold:
            self.clusters[best_cluster].append((board.copy(), value))
            self.position_clusters[position_hash] = best_cluster
        else:
            # Otherwise, create a new cluster
            new_cluster_id = self.cluster_count
            self.cluster_count += 1
            self.clusters[new_cluster_id].append((board.copy(), value))
            self.position_clusters[position_hash] = new_cluster_id
            self.stats['clusters_created'] += 1
        
        self.stats['positions_added'] += 1
        
        # If cache is too large, remove oldest entries (LRU policy)
        if len(self.cache) > self.cache_size:
            # Remove the first item (oldest) from the OrderedDict
            _, _ = self.cache.popitem(last=False)
    
    def _get_position_hash(self, board):
        """
        Create a simple hash of the board position.
        
        Args:
            board (chess.Board): The board position
            
        Returns:
            str: A hash string representing the position
        """
        # Use the FEN string without move counters as a hash
        fen_parts = board.fen().split(' ')
        return ' '.join(fen_parts[:4])
    
    def _calculate_similarity(self, board1, board2):
        """
        Calculate similarity between two board positions.
        
        Args:
            board1 (chess.Board): First board position
            board2 (chess.Board): Second board position
            
        Returns:
            float: Similarity score (0-1)
        """
        # Convert boards to piece maps
        pieces1 = board1.piece_map()
        pieces2 = board2.piece_map()
        
        # Count matching pieces
        matches = 0
        total = max(len(pieces1), len(pieces2))
        
        if total == 0:
            return 1.0  # Empty boards are identical
        
        for square, piece1 in pieces1.items():
            if square in pieces2 and pieces2[square] == piece1:
                matches += 1
        
        return matches / total
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            dict: Dictionary of cache statistics
        """
        stats = self.stats.copy()
        stats['cache_size'] = len(self.cache)
        stats['cluster_count'] = self.cluster_count
        
        # Calculate hit rates
        total = stats['total_lookups']
        if total > 0:
            stats['exact_hit_rate'] = stats['hits'] / total
            stats['cluster_hit_rate'] = stats['cluster_hits'] / total
            stats['total_hit_rate'] = (stats['hits'] + stats['cluster_hits']) / total
        else:
            stats['exact_hit_rate'] = 0
            stats['cluster_hit_rate'] = 0
            stats['total_hit_rate'] = 0
        
        return stats
