from collections import defaultdict

LINES = [
    (0, 1, 2),  # Row 1
    (3, 4, 5),  # Row 2
    (6, 7, 8),  # Row 3
    (0, 3, 6),  # Column 1
    (1, 4, 7),  # Column 2
    (2, 5, 8),  # Column 3
    (0, 4, 8),  # Diagonal top-left to bottom-right
    (2, 4, 6)   # Diagonal top-right to bottom-left
]

def is_valid(assignment, available):
    for line in LINES:
        # Gather the values placed in the current line.
        values = [assignment[i] for i in line if assignment[i] is not None]
        count = len(values)
        s = sum(values)
        if count == 3:
            # The line is complete; it must sum to 15.
            if s != 15:
                return False
        elif count == 2:
            # One cell is missing; the missing number must be available.
            if (15 - s) not in available:
                return False
        elif count == 1:
            # Two cells are missing; check that some pair from available can complete the sum.
            sorted_avail = sorted(available)
            if len(sorted_avail) < 2:
                return False  # Should not happen.
            min_possible = sorted_avail[0] + sorted_avail[1]
            max_possible = sorted_avail[-1] + sorted_avail[-2]
            if s + min_possible > 15 or s + max_possible < 15:
                return False
        else:  # count == 0: All cells in this line are empty.
            sorted_avail = sorted(available)
            if len(sorted_avail) >= 3:
                min_possible = sorted_avail[0] + sorted_avail[1] + sorted_avail[2]
                max_possible = sorted_avail[-1] + sorted_avail[-2] + sorted_avail[-3]
                if s + min_possible > 15 or s + max_possible < 15:
                    return False
    return True

# Global dictionaries and counters.
branching = defaultdict(list)  # maps depth -> list of branching factors at that depth
node_count = 0                 # total unique nodes (states) visited
max_depth_reached = 0          # maximum depth reached

# Global set for duplicate elimination.
# We record each state as a key: (tuple(assignment), tuple(sorted(available)))
visited_states = set()

def search(assignment, available, depth=0):
    """
    Recursively enumerate the search tree (free placement) for the 3×3 magic square.
    - assignment: list of 9 entries (None if not yet assigned).
    - available: set of numbers not yet used.
    - depth: the number of moves made so far.
    """
    global node_count, max_depth_reached, visited_states

    # Create a key representing this state.
    key = (tuple(assignment), tuple(sorted(available)))
    if key in visited_states:
        return
    visited_states.add(key)

    node_count += 1
    if depth > max_depth_reached:
        max_depth_reached = depth

    # If no available numbers remain, this is a complete board.
    if len(available) == 0:
        branching[depth].append(0)
        return

    valid_moves = []
    # For each empty cell, try every available number.
    for cell in range(9):
        if assignment[cell] is None:
            for num in available:
                new_assignment = assignment.copy()
                new_assignment[cell] = num
                new_available = available.copy()
                new_available.remove(num)
                if is_valid(new_assignment, new_available):
                    valid_moves.append((cell, num, new_assignment, new_available))
                    
    # Record the branching factor at this node.
    branching[depth].append(len(valid_moves))
    
    # Recurse on each valid move.
    for (cell, num, new_assignment, new_available) in valid_moves:
        search(new_assignment, new_available, depth+1)

def compute_effective_branching_factor(N, D, tol=1e-6):
    """
    Compute the effective branching factor b* from:
    
         sum_{d=0}^{D} (b*)^d = N
         
    For b* neq 1, the series sums to ((b*)^(D+1) - 1) / (b* - 1).
    We solve this using binary search.
    """
    if N == D + 1:
        return 1.0

    def f(b):
        if abs(b - 1.0) < 1e-9:
            return (D + 1) - N
        return (b**(D+1) - 1) / (b - 1) - N

    lo = 1.0
    hi = 10.0
    while f(hi) < 0:
        hi *= 2
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0

if __name__ == '__main__':
    initial_assignment = [None] * 9
    available_numbers = set(range(1, 10))
    
    # Start the recursive search from the empty board.
    search(initial_assignment, available_numbers, depth=0)
    
    print("Average branching factor per depth (free placement):")
    for d in sorted(branching.keys()):
        avg = sum(branching[d]) / len(branching[d])
        print(f"  Depth {d:2}: avg branching factor = {avg:.2f} (from {len(branching[d])} nodes)")
    
    print("\nTotal unique states visited:", node_count)
    print("Maximum depth reached:", max_depth_reached)
    
    # --- Effective branching factor computation ---
    # Exclude the terminal depth because all solution nodes have 0 moves.
    if max_depth_reached > 0:
        nonterminal_depth = max_depth_reached - 1
    else:
        nonterminal_depth = 0

    # Compute total nodes in levels 0..nonterminal_depth.
    N_internal = sum(len(branching[d]) for d in branching if d <= nonterminal_depth)
    
    b_eff = compute_effective_branching_factor(N_internal, nonterminal_depth)
    print("Overall effective branching factor b* (excluding terminal solutions) = {:.4f}".format(b_eff))
