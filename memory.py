"""
Memory modules for experience replay in chess reinforcement learning.
"""

import random
import numpy as np
import torch
from collections import namedtuple, deque

# Define the Experience Replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done', 'mask', 'next_mask'))

class ReplayMemory:
    """Standard Replay Memory for storing and sampling experiences."""
    
    def __init__(self, capacity):
        """
        Initialize a replay memory with fixed capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """
        Save an experience to memory.
        
        Args:
            *args: Components of an experience (state, action, next_state, reward, done, mask, next_mask)
        """
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Memory.
    
    Implements prioritized experience replay as described in the paper:
    "Prioritized Experience Replay" by Schaul et al.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize a prioritized replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta_start: Initial value of beta for importance sampling
            beta_frames: Number of frames over which beta will be annealed from beta_start to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # For beta calculation
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # Initial max priority
    
    def beta_by_frame(self, frame_idx):
        """
        Calculate beta value for importance sampling.
        
        Beta linearly increases from beta_start to 1 over beta_frames frames.
        
        Args:
            frame_idx: Current frame number
            
        Returns:
            Current beta value
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, *args):
        """
        Save an experience with maximum priority.
        
        Args:
            *args: Components of an experience (state, action, next_state, reward, done, mask, next_mask)
        """
        # Use max priority for new experiences
        priority = self.max_priority
        
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
        else:
            self.memory[self.position] = Experience(*args)
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences with prioritization.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (samples, indices, weights) where:
                - samples: List of sampled experiences
                - indices: Indices of sampled experiences
                - weights: Importance sampling weights
        """
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled indices.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)
