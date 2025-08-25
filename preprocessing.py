import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import time
import psutil
import os
import csv

# Define the 7 Soma cube pieces using coordinates
z = [(0,0,0),(1,0,0),(1,1,0),(2,1,0)]  # blue piece
p = [(0,0,0),(0,1,0),(0,1,1),(1,1,0)]  # red piece
t = [(0,0,0),(0,1,0),(1,1,0),(0,2,0)]  # purple piece
b = [(0,0,0),(1,0,0),(0,1,0),(0,1,1)]  # brown piece
a = [(0,0,0),(0,0,1),(0,1,0),(1,1,0)]  # yellow piece
l = [(0,0,0),(1,0,0),(2,0,0),(0,1,0)]  # orange piece
v = [(0,0,0),(1,0,0),(0,1,0)]          # green piece

pieces = [z, p, t, b, a, l, v]
colors = ["blue", "red", "purple", "brown", "yellow", "orange", "green"]

# Global statistics and state counters
nodes_visited = 0
num_backtracks = 0
backtrack_depths = defaultdict(int)
nodes_at_depth = defaultdict(int)
children_at_depth = defaultdict(int)
first_solution_nodes = None
solutions_found = 0  # For progress tracking

def reset_statistics():
    """Reset all solving statistics (and progress counter)"""
    global nodes_visited, num_backtracks, backtrack_depths, nodes_at_depth, children_at_depth, first_solution_nodes, solutions_found
    nodes_visited = 0
    num_backtracks = 0
    first_solution_nodes = None
    solutions_found = 0
    backtrack_depths.clear()
    nodes_at_depth.clear()
    children_at_depth.clear()

def calculate_branching_factors():
    """Calculate various branching factor metrics"""
    max_depth = max(nodes_at_depth.keys()) if nodes_at_depth else 0
    total_nodes = sum(nodes_at_depth.values())
    
    def total_nodes_with_bf(b):
        return sum(b**d for d in range(0, max_depth + 1)) - total_nodes
    
    if total_nodes == 0:
        return 0, 0, {}
    
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
    
    weighted_sum = sum(bf_by_depth[d] * nodes_at_depth[d] for d in range(max_depth))
    total_nodes_2 = sum(nodes_at_depth[d] for d in range(max_depth))
    average_bf = weighted_sum / total_nodes_2 if total_nodes_2 else 0
    
    return effective_bf, average_bf, bf_by_depth

def print_statistics():
    """Print solving statistics including branching factors"""
    print(f"\nSolving Statistics:")
    if first_solution_nodes is not None:
        print(f"Nodes visited to first solution: {first_solution_nodes}")
    print(f"Total nodes visited: {nodes_visited}")
    print(f"Total backtracks: {num_backtracks}")
    
    if nodes_at_depth:
        max_depth = max(nodes_at_depth.keys())
        print("\nDiagnostic Information:")
        print("Children at each depth:", dict(children_at_depth))
        print("Nodes at each depth:", dict(nodes_at_depth))
        print("Total children:", sum(children_at_depth.values()))
        print("Total internal nodes:", sum(nodes_at_depth[d] for d in range(max_depth-1)))
    
    effective_bf, average_bf, bf_by_depth = calculate_branching_factors()
    print(f"\nBranching Factor Analysis:")
    print(f"Effective branching factor (b*): {effective_bf:.2f}")
    print(f"Simple average branching factor: {average_bf:.2f}")
    
    if bf_by_depth:
        print("\nBranching factor by depth:")
        for depth, bf in bf_by_depth.items():
            print(f"Depth {depth}: {bf:.2f} (nodes: {nodes_at_depth[depth]})")
    
    # Plotting (optional)
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
        'font.family': 'Palatino'
    })
    
    if nodes_at_depth:
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
            total_backtracks_val = sum(backtrack_depths.values())
            backtrack_percentages = [backtrack_depths[d]/total_backtracks_val * 100 for d in depths_backtracks]
            ax2.bar(depths_backtracks, backtrack_percentages, 
                    color=colors_plot[1], alpha=0.8, edgecolor='white', linewidth=1.5)
            ax2.set_title('Backtrack Distribution by Depth', pad=20)
            ax2.set_xlabel('Depth in Search Tree')
            ax2.set_ylabel('Percentage of Total Backtracks')
            ax2.set_ylim(bottom=0)
            if max(backtrack_percentages) < 5:
                ax2.set_ylim(top=5)
        
        depths_bf = sorted(bf_by_depth.keys())
        bf_values = [bf_by_depth[d] for d in depths_bf]
        ax3.plot(depths_bf, bf_values, marker='o', color=colors_plot[2], linewidth=3, markersize=8)
        ax3.axhline(y=effective_bf, color=colors_plot[4], linestyle='--', linewidth=2,
                    label=f'Effective b* ({effective_bf:.2f})')
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
    """Plot a single solution using matplotlib"""
    color_map = {
        "blue": '#0000FF',
        "red": '#FF0000',
        "purple": '#800080',
        "brown": '#8B4513',
        "yellow": '#FFD700',
        "orange": '#FFA500',
        "green": '#008000'
    }
    for x, y, z, color in solution:
        if color:
            ax.bar3d(x, y, z, 1, 1, 1, color=color_map[color], shade=True, alpha=0.8)
    ax.view_init(elev=30, azim=45)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

# Rotation functions
rotate_x = lambda cubelets: [(x, z, -y) for (x, y, z) in cubelets]
rotate_y = lambda cubelets: [(z, y, -x) for (x, y, z) in cubelets]
rotate_z = lambda cubelets: [(-y, x, z) for (x, y, z) in cubelets]
identity = lambda cubelets: cubelets

def translate(piece):
    d_x, d_y, d_z = np.min(np.array(piece), axis=0) * -1
    return [(x + d_x, y + d_y, z + d_z) for (x, y, z) in piece]

def generate_rotations(piece):
    orientations = []
    for f_a in [identity, rotate_x, rotate_y, rotate_z]:
        for f_b in [identity, rotate_x, rotate_y, rotate_z]:
            for f_c in [identity, rotate_x, rotate_y, rotate_z]:
                for f_d in [identity, rotate_x, rotate_y, rotate_z]:
                    for f_e in [identity, rotate_x, rotate_y, rotate_z]:
                        rot_piece = sorted(f_a(f_b(f_c(f_d(f_e(piece))))))
                        min_x, min_y, min_z = rot_piece[0]
                        trans_rot_piece = [(x - min_x, y - min_y, z - min_z) for x, y, z in rot_piece]
                        if trans_rot_piece not in orientations:
                            orientations.append(trans_rot_piece)
    return orientations

# Generate all possible orientations for each piece
orientations = list(map(generate_rotations, pieces))
pieces_used = [False] * 7

#############################################
# Landmark Heuristics: Preprocessing & Query
#############################################

# Global dictionary to hold landmark information.
# Keys: canonical state (at depth==2)
# Now includes 'solutions_count' to track total complete solutions found below that landmark.
landmark_states = defaultdict(lambda: {
    'frequency': 0,
    'solutions': None,
    'state': None,
    'index': None,
    'solutions_count': 0
})

def canonical_state(solution, current_depth):
    """
    Return a canonical representation of the current state.
    We use:
      - current_depth (now 2 in our use-case)
      - tuple of pieces_used (global state)
      - tuple of filled board cells (positions with non-None colors)
    """
    used = tuple(pieces_used)
    filled = tuple(sorted([(x, y, z, col) for (x, y, z, col) in solution if col is not None]))
    return (current_depth, used, filled)

def solve_soma_dfs(solution, i, mode='normal', selected_landmarks=None):
    """
    Modified DFS function.
      - mode: 'preprocess', 'query', or 'normal'.
      - In 'preprocess' mode, when current_depth==2 the state is recorded and its solution count is accumulated.
      - In 'query' mode, if selected_landmarks is provided and current_depth==2,
        then if the canonical state is in selected_landmarks, we jump using cached solutions.
    """
    global nodes_visited, num_backtracks, first_solution_nodes, solutions_found
    nodes_visited += 1
    current_depth = sum(1 for p in pieces_used if p)
    nodes_at_depth[current_depth] += 1

    key = None
    # Preprocessing: record landmarks at depth 2
    if mode == 'preprocess' and current_depth == 3:
        key = canonical_state(solution, current_depth)
        if key not in landmark_states:
            landmark_states[key] = {
                'frequency': 1,
                'solutions': None,
                'state': solution.copy(),
                'index': i,
                'solutions_count': 0
            }
        else:
            landmark_states[key]['frequency'] += 1

    # Query mode: jump via landmark if applicable at depth 2.
    if mode == 'query' and selected_landmarks is not None and current_depth == 3:
        key = canonical_state(solution, current_depth)
        if key in selected_landmarks:
            if landmark_states[key]['solutions'] is None:
                sols = solve_soma_dfs(solution, i, mode='query', selected_landmarks=None)
                landmark_states[key]['solutions'] = sols
            return landmark_states[key]['solutions']
    
    # Base Case: complete solution
    if i == 27:
        solutions_found += 1
        if solutions_found % 100 == 0:
            print(f"Progress: {solutions_found} solutions found (Nodes visited: {nodes_visited}).")
        if first_solution_nodes is None:
            first_solution_nodes = nodes_visited
        return [solution]
    
    solutions = []
    children_count = 0
    x, y, z, _ = solution[i]
    for piece in range(7):
        if not pieces_used[piece]:
            for orientation in orientations[piece]:
                empty_coords = [(x + d_x, y + d_y, z + d_z, None) for (d_x, d_y, d_z) in orientation]
                if all(tup in solution for tup in empty_coords):
                    children_count += 1
                    pieces_used[piece] = True
                    filled_coords = [(x + d_x, y + d_y, z + d_z, colors[piece]) for (d_x, d_y, d_z) in orientation]
                    new_solution = sorted([tup for tup in solution if tup not in empty_coords] + filled_coords)
                    j = i
                    while j < 27 and new_solution[j][3]:
                        j += 1
                    sols = solve_soma_dfs(new_solution, j, mode=mode, selected_landmarks=selected_landmarks)
                    solutions.extend(sols)
                    pieces_used[piece] = False
    children_at_depth[current_depth] += children_count

    # For landmarks at depth 2 in preprocess mode, accumulate complete solution count.
    if mode == 'preprocess' and current_depth == 3 and key is not None:
        landmark_states[key]['solutions_count'] += len(solutions)
    
    return solutions

#############################################
# Running the Preprocessing and Query Phase
#############################################

def run_preprocessing():
    """
    Run the DFS from the root in 'preprocess' mode to collect landmarks.
    Frequency values and solution counts are recorded for each landmark.
    """
    reset_statistics()
    global pieces_used
    pieces_used = [False] * 7
    cube_coordinates = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    initial_solution = sorted([(x, y, z, None) for x, y, z in cube_coordinates])
    all_solutions = solve_soma_dfs(initial_solution, 0, mode='preprocess')
    print(f"\nPreprocessing complete. Total solutions found: {len(all_solutions)}")
    print(f"Number of unique landmark states (at depth 2): {len(landmark_states)}")
    return all_solutions

def run_query_phase(selected_keys):
    """
    Run the DFS from the root in 'query' mode using a set of selected landmark keys.
    Returns the list of solutions and the total nodes visited (QueryOps).
    """
    reset_statistics()
    global pieces_used
    pieces_used = [False] * 7
    cube_coordinates = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    initial_solution = sorted([(x, y, z, None) for x, y, z in cube_coordinates])
    sols = solve_soma_dfs(initial_solution, 0, mode='query', selected_landmarks=selected_keys)
    return sols, nodes_visited

def print_results_table():
    """
    For each landmark set size (n), run the query phase and output a table with:
      - n
      - PreprocOps: total number of complete solutions that can be found using the top-n landmarks
      - QueryOps: total nodes visited in the query phase
      - SolutionsFound: number of solutions found in the query phase
      - MeanSteps/sol: QueryOps divided by SolutionsFound
    Landmarks are sorted based on frequency * solutions_count.
    """
    if not landmark_states:
        print("No landmarks were recorded during preprocessing.")
        return
    
    # Sort landmarks by the product of frequency and solutions_count.
    sorted_landmarks = sorted(
        landmark_states.items(),
        key=lambda x: x[1]['frequency'] * x[1]['solutions_count'],
        reverse=True
    )

    results = []
    for n in [10, 20, 30, 40, 50]:
        if len(sorted_landmarks) < n:
            print(f"Only {len(sorted_landmarks)} landmarks recorded; skipping n={n}.")
            continue

        # Select the top-n landmark keys.
        selected_keys = {key for key, _ in sorted_landmarks[:n]}

        # Now, PreprocOps is defined as the total number of solutions (from preprocessing)
        # that can be found using these landmarks.
        preproc_ops = sum(landmark_states[key]['solutions_count'] for key in selected_keys)

        # Run the query phase using these landmarks.
        sols, query_ops = run_query_phase(selected_keys)
        num_sols = len(sols)
        mean_steps = query_ops / num_sols if num_sols > 0 else float('inf')

        results.append((n, preproc_ops, query_ops, num_sols, mean_steps))

    # Print the results table.
    print("\nLandmark Query Phase Results:")
    print(" n    PreprocOps   QueryOps   SolutionsFound   MeanSteps/sol")
    for (n, preproc_ops, query_ops, sols_found, mean) in results:
        print(f"{n:<5} {preproc_ops:<12} {query_ops:<10} {sols_found:<16} {mean:.2f}")

#############################################
# The Original Solver (for plotting, etc.)
#############################################

def solve_soma(coordinates, num_solutions_to_plot=10):
    """
    Solve the puzzle in 'normal' mode (without landmark shortcuts).
    """
    global pieces_used
    pieces_used = [False] * 7
    reset_statistics()
    solution = sorted([(x, y, z, None) for x, y, z in coordinates])
    all_solutions = solve_soma_dfs(solution, 0, mode='normal')
    
    print(f"\nTotal solutions found: {len(all_solutions)}")
    print_statistics()
    
    if num_solutions_to_plot > 0 and len(all_solutions) > 0:
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (20, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'figure.facecolor': 'white',
            'font.family': ['Palatino', 'serif']
        })
        fig = plt.figure()
        for i in range(min(num_solutions_to_plot, len(all_solutions))):
            ax = fig.add_subplot(2, 5, i+1, projection='3d')
            plot_solution(all_solutions[i], ax)
            ax.set_title(f'Solution {i+1}', pad=15)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()
    
    return all_solutions

def is_valid_position(coords, cube_size=3):
    """Check if piece coordinates are within cube bounds and are non-negative."""
    return all(0 <= x < cube_size and 0 <= y < cube_size and 0 <= z < cube_size for x, y, z in coords)

def track_first_moves(solution, i):
    """Modified version that only tracks first moves."""
    first_moves = []
    x, y, z, _ = solution[i]
    for piece in range(7):
        if not pieces_used[piece]:
            for orientation in orientations[piece]:
                empty_coords = [(x + d_x, y + d_y, z + d_z, None) for (d_x, d_y, d_z) in orientation]
                if all(tup in solution for tup in empty_coords):
                    filled_coords = [(x + d_x, y + d_y, z + d_z, colors[piece]) for (d_x, d_y, d_z) in orientation]
                    new_solution = sorted([tup for tup in solution if tup not in empty_coords] + filled_coords)
                    first_moves.append(new_solution)
    return first_moves

def plot_dfs_first_moves():
    """Plot the first 55 moves that DFS attempts."""
    global pieces_used
    pieces_used = [False] * 7
    cube_coordinates = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    initial_solution = sorted([(x, y, z, None) for x, y, z in cube_coordinates])
    first_moves = track_first_moves(initial_solution, 0)
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Palatino',
        'font.serif': ['Palatino', 'Palatino Linotype', 'DejaVu Serif']
    })
    fig = plt.figure(figsize=(22, 10))
    for i, move in enumerate(first_moves):
        if i >= 55:
            break
        ax = fig.add_subplot(5, 11, i + 1, projection='3d')
        plot_solution(move, ax)
        piece_color = next(coord[3] for coord in move if coord[3] is not None)
        ax.set_title(f'Move {i+1}: {piece_color}', fontsize=8)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.show()

#############################################
#              Main Function
#############################################

def main():
    # (Optional) Visualize first DFS moves:
    print("Plotting first 55 DFS moves...")
    plot_dfs_first_moves()
    
    # Solve the complete puzzle without landmark shortcuts (optional)
    print("\nSolving complete puzzle without landmark heuristics...")
    cube_coordinates = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    _ = solve_soma(cube_coordinates)
    
    # Landmark Preprocessing Phase
    print("\nRunning Preprocessing Phase (recording landmark states at depth 2)...")
    _ = run_preprocessing()
    
    # Landmark Query Phase & Results Table
    print("\nRunning Query Phase with landmark shortcuts...")
    print_results_table()
    
if __name__ == "__main__":
    main()
