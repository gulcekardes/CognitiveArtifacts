import argparse
import random
from collections import defaultdict
import hashlib
import pickle

import numpy as np
import matplotlib.pyplot as plt

z = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]  
p = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 0)]  
t = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0)]  
b = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1)]  
a = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0)]  
l = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0)]  
v = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]       

pieces = [z, p, t, b, a, l, v]
colors = ["blue", "red", "purple", "brown", "yellow", "orange", "green"]

color_to_piece = {color: piece for piece, color in enumerate(colors)}

# Global statistics
nodes_visited = 0
num_backtracks = 0
backtrack_depths = defaultdict(int)
nodes_at_depth = defaultdict(int)
children_at_depth = defaultdict(int)
first_solution_nodes = None
solutions_found = 0

# Global flags for pruning and candidate (value) ordering
ENABLE_PRUNING = False
USE_FORWARD_CHECKING_ORDERING = False  # Forward-checking value ordering flag

# --- Global variables for dead state mode ---
DEAD_STATE_MODE = None
DEAD_STATE_HASHES = None

# --- Global variables for biased value ordering ---
USE_BIASED_VALUE_ORDERING = False
PREPROCESS_BIAS_LIMIT = None    # When not None, we are in preprocessing mode and stop after a set number of solutions.
PREPROCESS_BIAS_DONE = False    # Flag to stop early during preprocessing.
value_bias = {}  # Dictionary: key=(x,y,z), value={piece: frequency, ...}

def reset_statistics():
    global nodes_visited, num_backtracks, backtrack_depths, nodes_at_depth, children_at_depth, first_solution_nodes, solutions_found
    nodes_visited = 0
    num_backtracks = 0
    first_solution_nodes = None
    solutions_found = 0
    backtrack_depths.clear()
    nodes_at_depth.clear()
    children_at_depth.clear()

def calculate_branching_factors():
    max_depth = max(nodes_at_depth.keys()) if nodes_at_depth else 0
    total_nodes = sum(nodes_at_depth.values())

    def total_nodes_with_bf(b):
        return sum(b ** d for d in range(0, max_depth + 1)) - total_nodes

    left, right = 1.0, 100.0
    while right - left > 0.0001:
        mid = (left + right) / 2
        if total_nodes_with_bf(mid) < 0:
            left = mid
        else:
            right = mid
    effective_bf = (left + right) / 2

    bf_by_depth = {}
    for depth in range(max_depth):
        nodes = nodes_at_depth[depth]
        children = children_at_depth[depth]
        bf_by_depth[depth] = children / max(1, nodes)

    weighted_sum = sum(bf_by_depth[d] * nodes_at_depth[d] for d in range(max_depth)) if max_depth > 0 else 0
    total_nodes_nonleaf = sum(nodes_at_depth[d] for d in range(max_depth)) if max_depth > 0 else 1
    average_bf = weighted_sum / total_nodes_nonleaf if total_nodes_nonleaf else 0.0

    return effective_bf, average_bf, bf_by_depth

def print_statistics():
    print(f"\nSolving Statistics:")
    if first_solution_nodes is not None:
        print(f"Nodes visited to first solution: {first_solution_nodes}")
    print(f"Total nodes visited: {nodes_visited}")
    print(f"Total backtracks: {num_backtracks}")

    if not nodes_at_depth:
        print("\n(No node-depth stats to report.)")
        return

    max_depth = max(nodes_at_depth.keys())

    print("\nDiagnostic Information:")
    print("Children at each depth:", dict(children_at_depth))
    print("Nodes at each depth:", dict(nodes_at_depth))
    print("Total children:", sum(children_at_depth.values()))
    print("Total internal nodes:", sum(nodes_at_depth[d] for d in range(max_depth - 1)) if max_depth >= 1 else 0)

    effective_bf, average_bf, bf_by_depth = calculate_branching_factors()
    print(f"\nBranching Factor Analysis:")
    print(f"Effective branching factor (b*): {effective_bf:.2f}")
    print(f"Simple average branching factor: {average_bf:.2f}")

    print("\nBranching factor by depth:")
    for depth, bf in bf_by_depth.items():
        nodes = nodes_at_depth[depth]
        print(f"Depth {depth}: {bf:.2f} (nodes: {nodes})")

    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (16, 12),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#f0f0f0',
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'DejaVu Sans',
        'font.serif': ['DejaVu Sans', 'DejaVu Sans Linotype', 'DejaVu Serif']
    })

    colors_plot = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F1C40F', '#9B59B6', '#1ABC9C']

    fig = plt.figure()
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    depths_nodes = sorted(nodes_at_depth.keys())
    ax1.bar(depths_nodes, [nodes_at_depth[d] for d in depths_nodes],
            color=colors_plot[0], alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.set_title('Node Visits Distribution by Depth', pad=20)
    ax1.set_xlabel('Depth in Search Tree')
    ax1.set_ylabel('Number of Nodes Visited')
    ax1.set_yscale('log')

    if backtrack_depths:
        depths_backtracks = sorted(backtrack_depths.keys())
        total_backtracks = sum(backtrack_depths.values())
        backtrack_percentages = [backtrack_depths[d] / total_backtracks * 100 for d in depths_backtracks] if total_backtracks else []
        ax2.bar(depths_backtracks, backtrack_percentages,
                color=colors_plot[1], alpha=0.8, edgecolor='white', linewidth=1.5)
        ax2.set_title('Backtrack Distribution by Depth', pad=20)
        ax2.set_xlabel('Depth in Search Tree')
        ax2.set_ylabel('Percentage of Total Backtracks')
        ax2.set_ylim(bottom=0)
        if backtrack_percentages and max(backtrack_percentages) < 5:
            ax2.set_ylim(top=5)

    ef_bf, _, depth_bf = calculate_branching_factors()
    depths_bf = sorted(depth_bf.keys())
    bf_values = [depth_bf[d] for d in depths_bf]
    ax3.plot(depths_bf, bf_values, marker='o', color=colors_plot[2], linewidth=3, markersize=8)
    ax3.axhline(y=ef_bf, color=colors_plot[4], linestyle='--', linewidth=2,
                label=f'Effective b* ({ef_bf:.2f})')
    ax3.set_title('Branching Factor Analysis by Depth', pad=20)
    ax3.set_xlabel('Depth in Search Tree')
    ax3.set_ylabel('Branching Factor')
    ax3.legend(frameon=True, facecolor='white', edgecolor='none')

    cumulative_nodes = np.cumsum([nodes_at_depth[d] for d in depths_nodes])
    ax4.plot(depths_nodes, cumulative_nodes, marker='o', color=colors_plot[5],
             linewidth=3, markersize=8)
    ax4.set_title('Cumulative Nodes by Depth', pad=20)
    ax4.set_xlabel('Depth in Search Tree')
    ax4.set_ylabel('Cumulative Number of Nodes')

    plt.show()

def plot_solution(solution, ax):
    color_map = {
        "blue": '#0000FF',
        "red": '#FF0000',
        "purple": '#800080',
        "brown": '#8B4513',
        "yellow": '#FFD700',
        "orange": '#FFA500',
        "green": '#008000'
    }
    for (x, y, z), color in solution.items():
        if color:
            ax.bar3d(x, y, z, 1, 1, 1, color=color_map[color], shade=True, alpha=0.8)
    ax.view_init(elev=30, azim=45)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def check_prunable_voids(solution):
    visited = set()
    for cell in solution:
        if solution[cell] is None and cell not in visited:
            comp = []
            stack = [cell]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                comp.append(current)
                for dx, dy, dz in [(1,0,0), (-1,0,0),
                                   (0,1,0), (0,-1,0),
                                   (0,0,1), (0,0,-1)]:
                    neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
                    if neighbor in solution and solution[neighbor] is None and neighbor not in visited:
                        stack.append(neighbor)
            if len(comp) < 3:
                return False
    return True

def rotate_identity(cubelets):
    """Return the piece unchanged."""
    return [(x, y, z) for (x, y, z) in cubelets]

def rotate_x_90(cubelets):
    """Rotate +90° around the X-axis: (x, y, z) -> (x, z, -y)."""
    return [(x, z, -y) for (x, y, z) in cubelets]

def rotate_y_90(cubelets):
    """Rotate +90° around the Y-axis: (x, y, z) -> (z, y, -x)."""
    return [(z, y, -x) for (x, y, z) in cubelets]

def rotate_z_90(cubelets):
    """Rotate +90° around the Z-axis: (x, y, z) -> (-y, x, z)."""
    return [(-y, x, z) for (x, y, z) in cubelets]

def normalize_to_origin(piece):
    """
    Shift a piece so its minimum (x,y,z) is at (0,0,0).
    Keeps relative geometry; used for canonicalizing orientations.
    """
    d_x, d_y, d_z = np.min(np.array(piece), axis=0) * -1
    return [(x + d_x, y + d_y, z + d_z) for (x, y, z) in piece]

_ROT_FNS = [rotate_identity, rotate_x_90, rotate_y_90, rotate_z_90]

def generate_unique_orientations(piece):
    """
    Produce all unique orientations of a piece by composing 90 axis rotations.
    Deduplicate by normalizing to the origin and sorting.
    """
    orientations = []
    seen = set()
    for f1 in _ROT_FNS:
        for f2 in _ROT_FNS:
            for f3 in _ROT_FNS:
                for f4 in _ROT_FNS:
                    for f5 in _ROT_FNS:
                        rot_piece = f1(f2(f3(f4(f5(piece)))))
                        normalized = normalize_to_origin(sorted(rot_piece))
                        normalized_sorted = tuple(sorted(normalized))
                        if normalized_sorted not in seen:
                            seen.add(normalized_sorted)
                            orientations.append(list(normalized_sorted))
    return orientations

# Generate all possible orientations for each piece
PIECE_ORIENTATIONS = list(map(generate_unique_orientations, pieces))

# Precompute anchor-cell → candidate (piece, orientation) placements
CANDIDATES_BY_CELL = defaultdict(set)
coordinates = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
for x, y, z in coordinates:
    for piece_idx in range(7):
        for orient in PIECE_ORIENTATIONS[piece_idx]:
            # Try aligning each cubelet of the piece to the anchor cell (x,y,z)
            for base_x, base_y, base_z in orient:
                adjusted = tuple((x + ox - base_x, y + oy - base_y, z + oz - base_z)
                                 for (ox, oy, oz) in orient)
                if all(cell in coordinates for cell in adjusted):
                    CANDIDATES_BY_CELL[(x, y, z)].add((piece_idx, adjusted))

def update_remaining(remaining, candidate):
    piece, orientation = candidate
    new_remaining = {}
    for cell, options in remaining.items():
        new_options = {(p, ori) for p, ori in options if p != piece and all(c not in ori for c in orientation)}
        new_remaining[cell] = new_options
    return new_remaining

# --- Forward Checking Value Ordering Functions ---
def forward_checking_score(candidate, solution, remaining):
    piece, orientation = candidate
    new_solution = solution.copy()
    for cell in orientation:
        new_solution[cell] = colors[piece]
    new_remaining = update_remaining(remaining, candidate)
    unassigned = [cell for cell in new_remaining if new_solution[cell] is None]
    for cell in unassigned:
        if len(new_remaining[cell]) == 0:
            return float('inf')
    score = sum(len(new_remaining[cell]) for cell in unassigned)
    return score

def order_candidates_with_forward_checking(candidates, solution, remaining):
    scored_candidates = []
    for candidate in candidates:
        score = forward_checking_score(candidate, solution, remaining)
        scored_candidates.append((score, candidate))
    scored_candidates.sort(key=lambda x: x[0])
    scored_candidates.reverse()  # higher score means more flexibility
    return [candidate for score, candidate in scored_candidates]

# --- Biased value ordering based on precomputed statistics ---
def biased_ordering(candidates, cell, bias_table):
    # For the given cell, sort candidates by frequency (descending)
    # Each candidate is a tuple: (piece, orientation)
    return sorted(candidates, key=lambda candidate: -bias_table.get(cell, {}).get(candidate[0], 0))

def update_bias(solution):
    # Update the global value_bias using the found solution.
    # solution is a dict mapping cell -> color.
    global value_bias
    for cell, color in solution.items():
        if color is not None:
            piece = color_to_piece[color]
            value_bias[cell][piece] += 1

# --- Modified Solver (recursive backtracking) ---
def solve_soma_(solution, remaining, get_next_cell):
    global nodes_visited, num_backtracks, first_solution_nodes, solutions_found
    global PREPROCESS_BIAS_DONE, PREPROCESS_BIAS_LIMIT

    # If in biased mode and preprocessing is done (only relevant during preprocessing), stop further search.
    if USE_BIASED_VALUE_ORDERING and PREPROCESS_BIAS_DONE:
        return []

    nodes_visited += 1
    current_depth = sum(pieces_used)  # Number of pieces placed so far.
    nodes_at_depth[current_depth] += 1

    # --- In query mode, prune if this state is known to be dead ---
    if DEAD_STATE_MODE == "query" and current_depth in (2, 3, 4, 5) and DEAD_STATE_HASHES is not None:
        state_hash = state_to_hash(solution, pieces_used)
        if state_hash in DEAD_STATE_HASHES.get(current_depth, set()):
            num_backtracks += 1
            backtrack_depths[current_depth] += 1
            return []

    if ENABLE_PRUNING and not check_prunable_voids(solution):
        num_backtracks += 1
        backtrack_depths[current_depth] += 1
        return []

    if all(color is not None for color in solution.values()):
        solutions_found += 1
        # During preprocessing mode, update bias and terminate early after enough solutions.
        if USE_BIASED_VALUE_ORDERING and PREPROCESS_BIAS_LIMIT is not None:
            update_bias(solution)
            if solutions_found >= PREPROCESS_BIAS_LIMIT:
                PREPROCESS_BIAS_DONE = True
                return []
        else:
            if solutions_found % 100 == 0:
                print(f'solutions_found={solutions_found}')
        if first_solution_nodes is None:
            first_solution_nodes = nodes_visited
        return [dict(sorted(solution.items()))]

    open_cells = [cell for cell, color in solution.items() if color is None]
    x, y, z = get_next_cell(open_cells, remaining)
    candidates = list(remaining[(x, y, z)])

    # --- Apply biased ordering if enabled; otherwise use forward checking or default ordering.
    if USE_BIASED_VALUE_ORDERING:
        candidates = biased_ordering(candidates, (x, y, z), value_bias)
    elif USE_FORWARD_CHECKING_ORDERING:
        candidates = order_candidates_with_forward_checking(candidates, solution, remaining)
    else:
        candidates = sorted(candidates, key=lambda item: item[0])

    children_count = 0
    solutions = []
    for piece, orientation in candidates:
        if pieces_used[piece]:
            continue
        children_count += 1
        pieces_used[piece] = True
        new_solution = solution.copy()
        valid = True
        for cell in orientation:
            if new_solution.get(cell) is not None:
                valid = False
                break
            new_solution[cell] = colors[piece]
        if not valid:
            pieces_used[piece] = False
            continue
        new_remaining = {cell: {(p, ori) for p, ori in remaining[cell] if p != piece and all(c not in ori for c in orientation)}
                         for cell in remaining.keys()}
        solutions.extend(solve_soma_(new_solution, new_remaining, get_next_cell))
        if not solutions:
            num_backtracks += 1
            depth = sum(pieces_used)
            backtrack_depths[depth] += 1
        pieces_used[piece] = False
    children_at_depth[current_depth] += children_count

    # --- In precompute mode, record states that yielded no solution ---
    if not solutions and DEAD_STATE_MODE == "precompute" and current_depth in (2, 3, 4, 5):
        state_hash = state_to_hash(solution, pieces_used)
        if current_depth not in DEAD_STATE_HASHES:
            DEAD_STATE_HASHES[current_depth] = set()
        DEAD_STATE_HASHES[current_depth].add(state_hash)
    return solutions

def solve_soma(coordinates, get_next_cell, num_solutions_to_plot=10):
    global pieces_used
    pieces_used = [False] * 7
    reset_statistics()
    solution = {(x, y, z): None for x, y, z in coordinates}
    all_solutions = solve_soma_(solution, CANDIDATES_BY_CELL, get_next_cell)
    print(f"\nTotal solutions found: {len(all_solutions)}")
    print_statistics()
    if num_solutions_to_plot > 0 and all_solutions:
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (20, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'figure.facecolor': 'white',
            'font.family': ['DejaVu Sans', 'serif']
        })
        fig = plt.figure()
        for i in range(min(num_solutions_to_plot, len(all_solutions))):
            ax = fig.add_subplot(2, 5, i + 1, projection='3d')
            plot_solution(all_solutions[i], ax)
            ax.set_title(f'Solution {i + 1}', pad=15)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()
    return all_solutions

def state_to_hash(solution, pieces_used):
    """Convert the current partial solution and pieces_used list into a unique hash."""
    sorted_solution = tuple(sorted(solution.items()))
    used = tuple(pieces_used)
    state_str = str(sorted_solution) + str(used)
    return hashlib.md5(state_str.encode()).hexdigest()

def is_valid_position(coords, cube_size=3):
    return all(0 <= x < cube_size and 0 <= y < cube_size and 0 <= z < cube_size for x, y, z in coords)

def track_first_moves(solution, i):
    first_moves = []
    # Here solution is a sorted list of tuples (x, y, z, color)
    x, y, z, _ = solution[i]
    for piece in range(7):
        if not pieces_used[piece]:
            for orientation in PIECE_ORIENTATIONS[piece]:
                empty_coords = [(x + d_x, y + d_y, z + d_z, None) for (d_x, d_y, d_z) in orientation]
                if all(tup in solution for tup in empty_coords):
                    filled_coords = [(x + d_x, y + d_y, z + d_z, colors[piece]) for (d_x, d_y, d_z) in orientation]
                    new_solution = sorted([tup for tup in solution if tup not in empty_coords] + filled_coords)
                    first_moves.append(new_solution)
    return first_moves

def plot_dfs_first_moves():
    global pieces_used
    pieces_used = [False] * 7
    cube_coordinates = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    initial_solution = sorted([(x, y, z, None) for x, y, z in cube_coordinates])
    first_moves = track_first_moves(initial_solution, 0)
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.serif': ['DejaVu Sans', 'DejaVu Sans Linotype', 'DejaVu Serif']
    })
    fig = plt.figure(figsize=(22, 10))
    for i, move in enumerate(first_moves):
        if i >= 55:
            break
        ax = fig.add_subplot(5, 11, i + 1, projection='3d')
        # Convert list-of-tuples back to dict for plotting
        move_dict = {(x, y, z): color for (x, y, z, color) in move}
        plot_solution(move_dict, ax)
        ax.view_init(elev=30, azim=45)
        piece_color = next(coord[3] for coord in move if coord[3] is not None)
        ax.set_title(f'Move {i + 1}: {piece_color}', fontsize=8)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.show()

def main():
    global ENABLE_PRUNING, USE_FORWARD_CHECKING_ORDERING, DEAD_STATE_MODE, DEAD_STATE_HASHES
    global USE_BIASED_VALUE_ORDERING, PREPROCESS_BIAS_LIMIT, value_bias, PREPROCESS_BIAS_DONE

    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_choice", default='deterministic',
                        choices=['deterministic', 'random', 'flexibility', 'MCV', 'layer_based', 'layer_mrv'])
    parser.add_argument("--weight_power", type=float, default=2, help='only used for flexibility ordering')
    parser.add_argument("--alpha", type=float, default=2.0, help="Weight for the layer (z-coordinate) in the layer_mrv heuristic")
    parser.add_argument("--pruning", action="store_true", help="Enable dynamic pruning of small voids (regions of size 1 or 2)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the randomized solver")
    parser.add_argument("--forward_checking_ordering", action="store_true", help="Enable forward checking value ordering")
    # --- NEW: Argument to activate dead state processing ---
    parser.add_argument("--dead_state_mode", choices=["precompute", "query"],
                        help="Activate dead state mode. 'precompute' runs the solver to record dead states (for depths 2,3,4,5) and saves them; 'query' loads precomputed dead states and prunes them during search.")
    # --- NEW: Arguments for biased value ordering ---
    parser.add_argument("--biased_value_ordering", action="store_true", help="Enable biased value ordering based on preprocessed statistics")
    parser.add_argument("--preprocess_bias_solutions", type=int, default=50, help="Number of solutions to precompute bias statistics")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    ENABLE_PRUNING = args.pruning
    USE_FORWARD_CHECKING_ORDERING = args.forward_checking_ordering

    # --- Setup dead state mode if activated ---
    if args.dead_state_mode is not None:
        DEAD_STATE_MODE = args.dead_state_mode
        if DEAD_STATE_MODE == "query":
            try:
                with open("dead_states.pkl", "rb") as f:
                    DEAD_STATE_HASHES = pickle.load(f)
                print("Loaded precomputed dead states from dead_states.pkl.")
            except FileNotFoundError:
                print("Error: dead_states.pkl not found. Run with --dead_state_mode precompute first.")
                return
        elif DEAD_STATE_MODE == "precompute":
            DEAD_STATE_HASHES = {2: set(), 3: set(), 4: set(), 5: set()}

    if args.selection_choice == 'deterministic':
        get_next_cell = lambda open_cells, rem: min(open_cells)
    elif args.selection_choice == 'random':
        get_next_cell = lambda open_cells, rem: random.choice(open_cells)
    elif args.selection_choice == 'flexibility':
        get_next_cell = lambda open_cells, rem: random.choices(
            open_cells, weights=[1 / (0.0001 + len(rem[cell]) ** args.weight_power) for cell in open_cells]
        )[0]
    elif args.selection_choice == 'MCV':
        get_next_cell = lambda open_cells, rem: random.choice(
            [cell for cell in open_cells if len(rem[cell]) == min(len(rem[c]) for c in open_cells)]
        )
    elif args.selection_choice == 'layer_based':
        get_next_cell = lambda open_cells, rem: min(
            [cell for cell in open_cells if cell[2] == min(c[2] for c in open_cells)],
            key=lambda cell: (len(rem[cell]), cell[0], cell[1])
        )
    elif args.selection_choice == 'layer_mrv':
        get_next_cell = lambda open_cells, rem: min(open_cells, key=lambda cell: (args.alpha * cell[2] + len(rem[cell]), cell[0], cell[1]))

    # print("Plotting first 55 DFS moves...")
    # plot_dfs_first_moves()

    cube_coordinates = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]

    # --- If biased value ordering is enabled, run a preprocessing phase to collect bias statistics ---
    if args.biased_value_ordering:
        USE_BIASED_VALUE_ORDERING = True
        PREPROCESS_BIAS_LIMIT = args.preprocess_bias_solutions
        PREPROCESS_BIAS_DONE = False
        # Initialize bias table: for each cell, create a frequency counter for each piece.
        value_bias = {(x, y, z): {piece: 0 for piece in range(7)} for x, y, z in cube_coordinates}
        print("Running preprocessing phase to compute bias statistics...")
        # Run the solver with num_solutions_to_plot=0 (no plotting) so that the recursion collects bias info.
        _ = solve_soma(cube_coordinates, get_next_cell, num_solutions_to_plot=0)
        print("Bias preprocessing complete. Collected bias statistics:")
        for cell in sorted(value_bias.keys()):
            print(f"Cell {cell}: {value_bias[cell]}")
        # Reset statistics and pieces for the main run.
        reset_statistics()
        pieces_used[:] = [False] * 7
        # --- FIX: Reset the early-termination flag and disable early termination during the main run.
        PREPROCESS_BIAS_LIMIT = None
        PREPROCESS_BIAS_DONE = False

    print("\nSolving complete puzzle...")
    solutions = solve_soma(cube_coordinates, get_next_cell)

    # --- In precompute mode, save the computed dead states and exit ---
    if DEAD_STATE_MODE == "precompute":
        with open("dead_states.pkl", "wb") as f:
            pickle.dump(DEAD_STATE_HASHES, f)
        print("Dead states precomputation complete and saved to dead_states.pkl.")

if __name__ == "__main__":
    # Global used by solver/trackers
    pieces_used = [False] * 7
    main()
