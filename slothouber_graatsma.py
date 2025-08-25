def idx(x, y, z):
    """Return the linear index for cell (x,y,z) in a 3×3×3 cube.
       We number cells 0..26 with index = x + 3*y + 9*z."""
    return x + 3 * y + 9 * z

def gen_placements():
    """Generate all 36 placements (as 4‐tuples of cell indices) for a 1×2×2 block 
       in a 3×3×3 cube (using only axis–aligned orientations)."""
    placements = []
    # Orientation: block’s short side along x (dimensions: 1×2×2)
    for x in range(3):
        for y in range(2):  # must have y and y+1 in {0,1,2}
            for z in range(2):  # must have z and z+1 in {0,1,2}
                placements.append((idx(x, y, z),
                                   idx(x, y+1, z),
                                   idx(x, y, z+1),
                                   idx(x, y+1, z+1)))
    # Orientation: block’s short side along y (dimensions: 2×1×2)
    for y in range(3):
        for x in range(2):  # x and x+1 in range(3)
            for z in range(2):
                placements.append((idx(x, y, z),
                                   idx(x+1, y, z),
                                   idx(x, y, z+1),
                                   idx(x+1, y, z+1)))
    # Orientation: block’s short side along z (dimensions: 2×2×1)
    for z in range(3):
        for x in range(2):
            for y in range(2):
                placements.append((idx(x, y, z),
                                   idx(x+1, y, z),
                                   idx(x, y+1, z),
                                   idx(x+1, y+1, z)))
    return placements

# Precompute the list of large-block placements.
large_placements = gen_placements()

def enumerate_states(max_depth=9):
    """
    Enumerate all distinct board states (and count moves) for 0 to max_depth moves.
    
    A state is represented as a triple:
       (occupancy, large_used, small_used)
    where occupancy is a 27‐tuple with values:
         0  : cell is empty
         1  : cell is occupied by a large block
         2  : cell is occupied by a small block
    large_used and small_used are the counts of blocks placed so far.
    
    Moves available from a state:
      - If large_used < 6, then for every placement in large_placements that
        fits in the empty cells, a move is possible.
      - If small_used < 3, then for every empty cell a small block move is possible.
    
    The function builds the state space level–by–level (by moves applied) and prints
    the average branching factor at each depth.
    """
    # The initial state: all cells empty, no blocks placed.
    initial_state = ((0,)*27, 0, 0)
    
    # Dictionary mapping depth -> set of state tuples.
    states_by_depth = {0: {initial_state}}
    
    # Dictionary to store average branching factors at each depth.
    branching_by_depth = {}
    
    for depth in range(max_depth):
        next_states = set()
        total_moves = 0
        count_states = 0
        
        for state in states_by_depth.get(depth, []):
            occupancy, large_used, small_used = state
            moves = 0  # count moves available from this state
            
            # Try placing a large block if allowed.
            if large_used < 6:
                for placement in large_placements:
                    # Check that all cells in the placement are empty.
                    if all(occupancy[cell] == 0 for cell in placement):
                        moves += 1
                        # Create a new occupancy list and mark the cells with 1.
                        occ_list = list(occupancy)
                        for cell in placement:
                            occ_list[cell] = 1
                        new_state = (tuple(occ_list), large_used + 1, small_used)
                        next_states.add(new_state)
            
            # Try placing a small block if allowed.
            if small_used < 3:
                for cell in range(27):
                    if occupancy[cell] == 0:
                        moves += 1
                        occ_list = list(occupancy)
                        occ_list[cell] = 2
                        new_state = (tuple(occ_list), large_used, small_used + 1)
                        next_states.add(new_state)
            
            total_moves += moves
            count_states += 1
        
        avg_branches = total_moves / count_states if count_states else 0
        branching_by_depth[depth] = avg_branches
        
        print(f"Depth {depth:2d}: States = {len(states_by_depth.get(depth, [])):6d}, "
              f"Total moves = {total_moves:6d}, Avg branching = {avg_branches:5.2f}")
        
        if next_states:
            states_by_depth[depth+1] = next_states
        else:
            break  # No further states to expand.
    
    return states_by_depth, branching_by_depth

def compute_effective_branching_factor(N, D, tol=1e-6):
    """
    Given:
      - N: total number of states (from depth 0 to D), and
      - D: maximum depth reached,
    compute the effective branching factor b* defined by:
    
         sum_{d=0}^{D} (b*)^d = N
         
    For b* ≠ 1, the geometric series sums to ((b*)^(D+1)-1)/(b* - 1).
    We solve the equation numerically via binary search.
    """
    # Trivial case: if N equals D+1 (only one state per level), then b* = 1.
    if N == D + 1:
        return 1.0

    def f(b):
        # Avoid division by zero when b is near 1.
        if abs(b - 1.0) < 1e-9:
            return (D + 1) - N
        return (b**(D+1) - 1) / (b - 1) - N

    lo = 1.0
    hi = 10.0  # initial high guess; adjust if needed.
    while f(hi) < 0:
        hi *= 2

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0

def state_to_str(state):
    """
    Convert a state (occupancy, large_used, small_used) to a compact string.
    We map:
      0 -> '.'
      1 -> 'L'
      2 -> 'S'
    The occupancy is a 27-tuple. We simply join the mapped characters.
    """
    occupancy, large_used, small_used = state
    mapping = {0: '.', 1: 'L', 2: 'S'}
    return ''.join(mapping[cell] for cell in occupancy)

def print_states_grid(states, rows, cols):
    """
    Print the given list of states (each represented as a string) in a grid of
    the specified number of rows and columns.
    """
    if len(states) != rows * cols:
        print(f"Warning: Expected {rows*cols} states, but got {len(states)}.")
    # For each row:
    for r in range(rows):
        # Collect the string representations for one row.
        row_cells = states[r*cols:(r+1)*cols]
        # Join with a separator.
        print(' | '.join(row_cells))
    print()  # extra blank line after grid

def count_moves(state):
    """
    Given a state (occupancy, large_used, small_used), compute the number
    of legal moves available from that state (using the same criteria as in enumeration).
    """
    occupancy, large_used, small_used = state
    moves = 0
    if large_used < 6:
        for placement in large_placements:
            if all(occupancy[cell] == 0 for cell in placement):
                moves += 1
    if small_used < 3:
        for cell in range(27):
            if occupancy[cell] == 0:
                moves += 1
    return moves

if __name__ == '__main__':
    # Enumerate all distinct states up to max_depth moves (here, max_depth=9).
    states_by_depth, branching_by_depth = enumerate_states(max_depth=9)
    
    # Compute total visited states.
    total_states = sum(len(states_by_depth[d]) for d in states_by_depth)
    max_depth_reached = max(states_by_depth.keys())
    
    print("\nTotal visited states:", total_states)
    print("Maximum depth reached:", max_depth_reached)
    
    # Compute overall effective branching factor b*
    b_eff = compute_effective_branching_factor(total_states, max_depth_reached)
    print("\nOverall effective branching factor b* = {:.4f}".format(b_eff))
    
    # --- EXTRA: Show all states at depth 8 from which moves are available ---
    # We want to see each move available at depth 8.
    # That is, for each state at depth 8, count its moves and repeat the state that many times.
    depth_to_inspect = 8
    if depth_to_inspect in states_by_depth:
        move_states = []
        for state in states_by_depth[depth_to_inspect]:
            m = count_moves(state)
            # Repeat the state once per available move.
            for _ in range(m):
                move_states.append(state)
        print(f"\nAt depth {depth_to_inspect}, the sum of moves over all states is {len(move_states)} (should be 60).")
        # Print them in a grid of 5 rows x 12 columns.
        if len(move_states) != 60:
            print("Note: The number of repeated move-states is not 60; adjusting grid dimensions accordingly.")
            import math
            cols = math.ceil(math.sqrt(len(move_states)))
            rows = math.ceil(len(move_states) / cols)
        else:
            rows, cols = 5, 12
        
        move_state_strs = [state_to_str(s) for s in move_states]
        print("\nStates at depth 8 (each repeated according to available moves):")
        print_states_grid(move_state_strs, rows, cols)
    else:
        print(f"No states found at depth {depth_to_inspect}.")
