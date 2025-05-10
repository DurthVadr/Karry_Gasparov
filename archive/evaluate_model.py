import chess
import chess.pgn
import io
import random
from model_integration import ModelIntegration

def evaluate_model_on_position(model, board, num_moves=5):
    """Evaluate a model on a specific position by looking at its next few moves"""
    print(f"Starting position:\n{board.unicode()}")
    
    for i in range(num_moves):
        if board.is_game_over():
            print(f"Game over: {board.result()}")
            break
            
        print(f"\nTurn {i+1}: {'White' if board.turn == chess.WHITE else 'Black'} to move")
        
        # Get move from model
        move = model.get_move(board)
        print(f"Model plays: {move.uci()} ({board.san(move)})")
        
        # Make the move
        board.push(move)
        print(f"Board after move:\n{board.unicode()}")
    
    return board

def compare_models(model_paths, num_positions=3, num_moves_per_position=5):
    """Compare multiple models on the same positions"""
    # Create model integration
    model_integration = ModelIntegration()
    
    # Load models
    models = []
    for path in model_paths:
        try:
            model = model_integration.load_model(path)
            print(f"Loaded model: {path} ({model.name})")
            models.append((path, model))
        except Exception as e:
            print(f"Error loading model {path}: {str(e)}")
    
    # Generate random positions by playing random moves
    positions = []
    for i in range(num_positions):
        board = chess.Board()
        # Play 5-15 random moves to get to a middle-game position
        num_random_moves = random.randint(5, 15)
        for _ in range(num_random_moves):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        positions.append(board.copy())
    
    # Evaluate each model on each position
    for i, board in enumerate(positions):
        print(f"\n{'='*50}")
        print(f"Position {i+1}")
        print(f"{'='*50}")
        
        for path, model in models:
            print(f"\n{'-'*30}")
            print(f"Evaluating model: {path}")
            print(f"{'-'*30}")
            
            # Create a copy of the board for this model
            board_copy = board.copy()
            
            # Evaluate the model
            evaluate_model_on_position(model, board_copy, num_moves=num_moves_per_position)

if __name__ == "__main__":
    # Models to compare
    model_paths = [
        "models/model_pgn_4000.pt",  # The model that was reported to be playing badly
        "models/model_final.pt",      # Another model for comparison
        # Add more models here if needed
    ]
    
    # Compare the models
    compare_models(model_paths, num_positions=2, num_moves_per_position=5)
