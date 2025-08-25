import chess
import random

def simulate_random_game(max_depth):
    board = chess.Board()
    branch_factors = []
    for d in range(max_depth + 1):
        legal_count = board.legal_moves.count()
        branch_factors.append(legal_count)
        if d == max_depth or board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    while len(branch_factors) < max_depth + 1:
        branch_factors.append(0)
    return branch_factors

def main():
    num_simulations = 10000
    max_depth = 40
    total_branch_counts = [0] * (max_depth + 1)

    for i in range(num_simulations):
        counts = simulate_random_game(max_depth)
        for d, count in enumerate(counts):
            total_branch_counts[d] += count
    print("Average Branching Factor Trends (sampled):")
    for d in range(max_depth + 1):
        avg = total_branch_counts[d] / num_simulations
        print(f"Depth {d:2d}: {avg:.2f}")

if __name__ == '__main__':
    main()
