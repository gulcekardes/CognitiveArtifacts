import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats

np.random.seed(42)
random.seed(42)

# Piece definitions
z = [(0,0,0),(1,0,0),(1,1,0),(2,1,0)]  # blue piece
p = [(0,0,0),(0,1,0),(0,1,1),(1,1,0)]  # red piece
t = [(0,0,0),(0,1,0),(1,1,0),(0,2,0)]  # purple piece
b = [(0,0,0),(1,0,0),(0,1,0),(0,1,1)]  # brown piece
a = [(0,0,0),(0,0,1),(0,1,0),(1,1,0)]  # yellow piece
l = [(0,0,0),(1,0,0),(2,0,0),(0,1,0)]  # orange piece
v = [(0,0,0),(1,0,0),(0,1,0)]          # green piece

pieces = [z, p, t, b, a, l, v]
colors = ["blue", "red", "purple", "brown", "yellow", "orange", "green"]

# Rotation functions
rotate_x = lambda cubelets: [(x, z, -y) for (x, y, z) in cubelets]
rotate_y = lambda cubelets: [(z, y, -x) for (x, y, z) in cubelets]
rotate_z = lambda cubelets: [(-y, x, z) for (x, y, z) in cubelets]
identity = lambda cubelets: cubelets

def plot_soma_state(state, used_pieces, title):
    """Plot a 3D visualization of a Soma cube state"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping with specific RGB values for better visibility
    color_map = {
        'blue': '#0000FF',
        'red': '#FF0000',
        'purple': '#800080',
        'brown': '#8B4513',
        'yellow': '#FFD700',
        'orange': '#FFA500',
        'green': '#008000'
    }

    def plot_cube(x, y, z, color):
        vertices = [
            [(x,y,z), (x+1,y,z), (x+1,y+1,z), (x,y+1,z)],  # bottom
            [(x,y,z+1), (x+1,y,z+1), (x+1,y+1,z+1), (x,y+1,z+1)],  # top
            [(x,y,z), (x+1,y,z), (x+1,y,z+1), (x,y,z+1)],  # front
            [(x,y+1,z), (x+1,y+1,z), (x+1,y+1,z+1), (x,y+1,z+1)],  # back
            [(x,y,z), (x,y+1,z), (x,y+1,z+1), (x,y,z+1)],  # left
            [(x+1,y,z), (x+1,y+1,z), (x+1,y+1,z+1), (x+1,y,z+1)]  # right
        ]
        ax.add_collection3d(Poly3DCollection(vertices, 
                                           facecolors=color_map[color],
                                           alpha=0.8,
                                           edgecolor='black'))

    # Plot each cube in the state
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if state[x,y,z] is not None:
                    plot_cube(x, y, z, state[x,y,z])

    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_zlim([0, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=20, azim=45)
    
    # Add title and used/available pieces information
    plt.title(title)
    used_text = "Used: " + ", ".join([colors[i] for i, u in enumerate(used_pieces) if u])
    available_text = "Available: " + ", ".join([colors[i] for i, u in enumerate(used_pieces) if not u])
    plt.figtext(0.05, 0.02, used_text + '\n' + available_text, wrap=True)
    
    return fig

def translate(piece):
    """Translate piece to origin"""
    d_x, d_y, d_z = np.min(np.array(piece), axis=0) * -1
    return [(x + d_x, y + d_y, z + d_z) for (x, y, z) in piece]

def generate_rotations(piece):
    """Generate all unique orientations of a piece"""
    orientations = []
    for f_a in [identity, rotate_x, rotate_y, rotate_z]:
        for f_b in [identity, rotate_x, rotate_y, rotate_z]:
            for f_c in [identity, rotate_x, rotate_y, rotate_z]:
                rot_piece = sorted(f_a(f_b(f_c(piece))))
                min_x, min_y, min_z = rot_piece[0]
                trans_rot_piece = [(x - min_x, y - min_y, z - min_z) for x, y, z in rot_piece]
                if trans_rot_piece not in orientations:
                    orientations.append(trans_rot_piece)
    return orientations

orientations = list(map(generate_rotations, pieces))

def get_state_signature(state, used_pieces):
    """Generate a unique signature for a given state"""
    pieces_list = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if state[x,y,z] is not None:
                    pieces_list.append((x,y,z,state[x,y,z]))
    return (tuple(sorted(pieces_list)), tuple(used_pieces))

def get_valid_moves(state, used_pieces):
    """Get all valid moves for a given state"""
    moves = []
    occupied_positions = set()
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if state[x,y,z] is not None:
                    occupied_positions.add((x,y,z))
    
    for piece_idx in range(7):
        if not used_pieces[piece_idx]:
            for orientation in orientations[piece_idx]:
                for x in range(3):
                    for y in range(3):
                        for z in range(3):
                            valid = True
                            new_positions = []
                            
                            for dx, dy, dz in orientation:
                                nx, ny, nz = x + dx, y + dy, z + dz
                                if not (0 <= nx < 3 and 0 <= ny < 3 and 0 <= nz < 3):
                                    valid = False
                                    break
                                if (nx, ny, nz) in occupied_positions:
                                    valid = False
                                    break
                                new_positions.append((nx, ny, nz))
                            
                            if valid:
                                moves.append((piece_idx, new_positions))
    return moves

def generate_exact_states():
    """Generate all states for depths 0-2"""
    empty_state = np.full((3,3,3), None)
    depth_0_state = {get_state_signature(empty_state, [False]*7): (empty_state, [False]*7)}
    
    depth_1_states = {}
    moves = get_valid_moves(empty_state, [False]*7)
    for piece_idx, positions in moves:
        new_state = empty_state.copy()
        new_used = [False]*7
        new_used[piece_idx] = True
        for px, py, pz in positions:
            new_state[px,py,pz] = colors[piece_idx]
        depth_1_states[get_state_signature(new_state, new_used)] = (new_state, new_used)
    
    depth_2_states = {}
    for state, used in depth_1_states.values():
        moves = get_valid_moves(state, used)
        for piece_idx, positions in moves:
            new_state = state.copy()
            new_used = used.copy()
            new_used[piece_idx] = True
            for px, py, pz in positions:
                new_state[px,py,pz] = colors[piece_idx]
            depth_2_states[get_state_signature(new_state, new_used)] = (new_state, new_used)
    
    return {
        0: (depth_0_state, len(get_valid_moves(empty_state, [False]*7))),
        1: (depth_1_states, [len(get_valid_moves(state, used)) for state, used in depth_1_states.values()]),
        2: (depth_2_states, [len(get_valid_moves(state, used)) for state, used in depth_2_states.values()])
    }

def sample_states_at_depth(depth, max_samples, exact_results):
    """Sample states at a given depth"""
    if depth <= 2:
        return exact_results[depth][0]
            
    states = {}
    prev_states = sample_states_at_depth(depth-1, max_samples, exact_results)
    
    # Randomly sample from the previous depth's states
    for prev_state_key in random.sample(list(prev_states.keys()), 
                                        min(len(prev_states), max_samples)):
        state, used = prev_states[prev_state_key]
        
        moves = get_valid_moves(state, used)
        if moves:
            piece_idx, positions = random.choice(moves)
            new_state = state.copy()
            new_used = used.copy()
            new_used[piece_idx] = True
            
            for px, py, pz in positions:
                new_state[px,py,pz] = colors[piece_idx]
            
            sig = get_state_signature(new_state, new_used)
            if sig not in states:
                states[sig] = (new_state, new_used)
                
            if len(states) >= max_samples:
                break
    
    return states

def plot_state_in_subplot(ax, state, used_pieces, title):
    """Helper to plot a single state in a given Axes3D subplot."""
    color_map = {
        'blue': '#0000FF',
        'red': '#FF0000',
        'purple': '#800080',
        'brown': '#8B4513',
        'yellow': '#FFD700',
        'orange': '#FFA500',
        'green': '#008000'
    }

    def plot_cube(x, y, z, color):
        vertices = [
            [(x,y,z), (x+1,y,z), (x+1,y+1,z), (x,y+1,z)],  
            [(x,y,z+1), (x+1,y,z+1), (x+1,y+1,z+1), (x,y+1,z+1)],  
            [(x,y,z), (x+1,y,z), (x+1,y,z+1), (x,y,z+1)],  
            [(x,y+1,z), (x+1,y+1,z), (x+1,y+1,z+1), (x,y+1,z+1)],  
            [(x,y,z), (x,y+1,z), (x,y+1,z+1), (x,y,z+1)],  
            [(x+1,y,z), (x+1,y+1,z), (x+1,y+1,z+1), (x+1,y,z+1)]  
        ]
        ax.add_collection3d(Poly3DCollection(vertices, 
                                           facecolors=color_map[color],
                                           alpha=0.8,
                                           edgecolor='black'))

    for x in range(3):
        for y in range(3):
            for z in range(3):
                if state[x,y,z] is not None:
                    plot_cube(x, y, z, state[x,y,z])

    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_zlim([0, 3])
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)

def plot_state_and_moves(state, used_pieces, moves, state_num):
    """Plot a state and all its possible next moves in a grid."""
    import math
    
    num_moves = len(moves)
    grid_size = math.ceil(math.sqrt(num_moves + 1))
    
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'State #{state_num} and its {num_moves} Possible Moves', fontsize=16, y=0.95)
    
    # Plot original state in first position
    ax = fig.add_subplot(grid_size, grid_size, 1, projection='3d')
    plot_state_in_subplot(ax, state, used_pieces, "Current State")
    
    # Plot each possible move
    for idx, (piece_idx, positions) in enumerate(moves, start=2):
        new_state = state.copy()
        new_used = used_pieces.copy()
        new_used[piece_idx] = True
        for px, py, pz in positions:
            new_state[px,py,pz] = colors[piece_idx]
        
        ax = fig.add_subplot(grid_size, grid_size, idx, projection='3d')
        plot_state_in_subplot(ax, new_state, new_used, 
                              f"Move {idx-1}: Add {colors[piece_idx]}")
    
    plt.tight_layout()
    plt.savefig(f'state_{state_num}_with_moves.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_grid_histograms_nonterminal(branching_distributions):
    """
    Create histograms for *non-terminal states only* (branching factor > 0),
    arranged one figure per depth.
    """
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['font.size'] = 19
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    for depth in range(1, 7):
        plt.figure(figsize=(8, 6), dpi=300)
        ax = plt.gca()
        
        if depth in branching_distributions and branching_distributions[depth]:
            data = branching_distributions[depth]
            data = np.array(data)
            non_zero_data = data[data > 0]  # focus on non-terminal states

            # Freedman-Diaconis or fallback to default bins
            try:
                q75, q25 = np.percentile(non_zero_data, [75, 25])
                iqr = q75 - q25
                n_bins = 20 if iqr == 0 else min(
                    50, 
                    max(20, int(np.ceil(
                        (max(non_zero_data) - min(non_zero_data)) / (2 * iqr / (len(non_zero_data) ** (1/3)))
                    )))
                )
            except:
                n_bins = 20
            
            # Plot histogram (proportions)
            weights = np.ones_like(non_zero_data) / len(data)
            plt.hist(non_zero_data, bins=n_bins, alpha=0.8,
                     color='#B2A5FF', edgecolor='#333333',
                     weights=weights)
            
            plt.xlabel('Branching Factor')
            plt.ylabel('Proportion of States')
            plt.title(f'Depth {depth} (Non-Terminal States)')
            
            ax.set_facecolor('#f0f0f0')
            plt.gca().patch.set_facecolor('#f0f0f0')
            plt.grid(True, linestyle='--', color='white', alpha=0.7)
            
            # Stats
            mean = np.mean(data)  # includes zeros
            std = np.std(data)
            terminal_frac = np.mean(data == 0)
            
            if len(non_zero_data) > 0:
                non_zero_mean = np.mean(non_zero_data)
                non_zero_std = np.std(non_zero_data)
                median = np.median(non_zero_data)
            else:
                non_zero_mean = non_zero_std = median = 0
                
            stats_text = (f'All States:\n'
                          f'μ = {mean:.2f}, σ = {std:.2f}\n'
                          f'Terminal: {terminal_frac:.1%}\n'
                          f'\nNon-Terminal Only:\n'
                          f'μ = {non_zero_mean:.2f}\n'
                          f'σ = {non_zero_std:.2f}\n'
                          f'median = {median:.2f}')
            
            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            plt.tight_layout()
            plt.savefig(f'branching_factors_depth_{depth}_nonterminal.pdf', 
                        dpi=300, bbox_inches='tight')
            plt.savefig(f'branching_factors_depth_{depth}_nonterminal.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()

def create_grid_histograms_including_terminal(branching_distributions):
    """
    Create a single figure with 6 subplots (depths 1 to 6),
    plotting the histogram of branching factors *including* zero.
    Each subplot shows mean and std in a small legend box.
    """
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    depths = [1, 2, 3, 4, 5, 6]
    
    for i, depth in enumerate(depths):
        ax = axes[i // 3, i % 3]
        
        if depth in branching_distributions and branching_distributions[depth]:
            data = np.array(branching_distributions[depth])  # includes zeros
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Freedman-Diaconis or fallback
            try:
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                n_bins = 20 if iqr == 0 else min(
                    50,
                    max(20, int(np.ceil(
                        (max(data) - min(data)) / (2 * iqr / (len(data) ** (1/3)))
                    )))
                )
            except:
                n_bins = 20

            # Plot histogram with proportions
            weights = np.ones_like(data) / len(data)
            ax.hist(data, bins=n_bins, alpha=0.8, color='#B2A5FF',
                    edgecolor='#333333', weights=weights)
            
            ax.set_title(f"Depth {depth}")
            ax.set_xlabel("Branching Factor")
            ax.set_ylabel("Proportion of States")
            ax.set_facecolor('#f0f0f0')
            ax.grid(True, linestyle='--', color='white', alpha=0.7)
            
            # Place mean and std in the top-right corner
            stats_text = f'μ={mean_val:.2f}, σ={std_val:.2f}'
            ax.text(0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            # No data available for this depth
            ax.set_title(f"Depth {depth}\nNo Data")
            ax.set_xlabel("Branching Factor")
            ax.set_ylabel("Proportion of States")
            ax.set_facecolor('#f0f0f0')
            ax.grid(True, linestyle='--', color='white', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('branching_factors_including_terminal.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig('branching_factors_including_terminal.pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_analysis(num_trials=5, samples_per_depth=10000):
    """Run analysis with both branching factor calculations."""
    exact_results = generate_exact_states()
    
    # Store results for *all* states (including zeros)
    depth_results_with_zero = defaultdict(list)
    
    # Initialize with exact results (depth 0,1,2)
    depth_results_with_zero[0].append(exact_results[0][1])
    depth_results_with_zero[1].extend(exact_results[1][1])
    depth_results_with_zero[2].extend(exact_results[2][1])
    
    print(f"\nRunning {num_trials} trials with {samples_per_depth} samples per depth...")
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        for depth in range(3, 7):
            states = sample_states_at_depth(depth, samples_per_depth, exact_results)
            if not states:
                break
            
            # Calculate branching factors (including zeros)
            branching_factors = [
                len(get_valid_moves(state, used)) 
                for state, used in states.values()
            ]
            depth_results_with_zero[depth].extend(branching_factors)
    
    # 1) Create your original non-terminal histograms (renamed function).
    create_grid_histograms_nonterminal(depth_results_with_zero)
    
    # 2) Create new 6-subplot figure including terminal states.
    create_grid_histograms_including_terminal(depth_results_with_zero)
    
    return depth_results_with_zero

if __name__ == "__main__":
    results = run_analysis()
