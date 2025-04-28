import chess
import chess.pgn
import random
import io
import os

def generate_random_game(max_moves=100):
    """Generate a random chess game"""
    board = chess.Board()
    game = chess.pgn.Game()
    
    # Set some game headers
    game.headers["Event"] = "Synthetic Game"
    game.headers["Site"] = "Synthetic Database"
    game.headers["Date"] = "2025.04.27"
    game.headers["Round"] = "1"
    game.headers["White"] = "Engine1"
    game.headers["Black"] = "Engine2"
    game.headers["Result"] = "*"
    
    node = game
    
    # Make random moves until the game is over or max_moves is reached
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        move = random.choice(legal_moves)
        board.push(move)
        node = node.add_variation(move)
        move_count += 1
    
    # Set the result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        result = "1/2-1/2"
    else:
        result = "*"
    
    game.headers["Result"] = result
    
    return game

def save_pgn(game, filename):
    """Save a game to a PGN file"""
    with open(filename, 'a') as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
        f.write("\n\n")  # Add some space between games

def generate_dataset(num_games=1000, output_file="synthetic_games.pgn"):
    """Generate a dataset of random chess games"""
    for i in range(num_games):
        game = generate_random_game()
        save_pgn(game, output_file)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} games")

if __name__ == "__main__":
    output_dir = "/home/ubuntu/data/synthetic"
    output_file = os.path.join(output_dir, "synthetic_games.pgn")
    
    # Generate 1000 random games
    print("Generating synthetic chess games...")
    generate_dataset(num_games=1000, output_file=output_file)
    print(f"Dataset generated and saved to {output_file}")
