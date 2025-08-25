from collections import deque, defaultdict

def get_neighbors(state):
    """
    Given a state (a tuple of 9 ints, with 0 representing the blank),
    return a list of states reachable by sliding an adjacent tile into the blank.
    This function simply returns all physically legal moves.
    """
    neighbors = []
    i = state.index(0)  # Find the index of the blank.
    r, c = divmod(i, 3)
    # Moves: up, down, left, right.
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            new_state = list(state)
            new_state[i], new_state[j] = new_state[j], new_state[i]
            neighbors.append(tuple(new_state))
    return neighbors

def bfs_enumeration_with_full_moves(start_state):
    """
    Enumerates the state space of the 8-puzzle using BFS.
    At each state, we record the full branching factor (i.e. the number
    of moves available from that state, regardless of whether the move goes back
    to the state from which we came).
    
    We use a visited set to avoid cycles.
    
    Returns:
      - branching: a dict mapping depth -> list of full branching factors at that depth.
      - node_count: the total number of unique states visited.
      - max_depth: the maximum depth reached in the search.
    """
    queue = deque([(start_state, 0)])  # Each element is a (state, depth) pair.
    visited = {start_state}
    branching = defaultdict(list)
    node_count = 0
    max_depth = 0

    while queue:
        state, depth = queue.popleft()
        node_count += 1
        if depth > max_depth:
            max_depth = depth

        # Record the full number of legal moves from this state.
        full_moves = get_neighbors(state)
        branching[depth].append(len(full_moves))

        # Enqueue only states we haven't seen yet to avoid infinite loops.
        for n in full_moves:
            if n not in visited:
                visited.add(n)
                queue.append((n, depth + 1))
                
    return branching, node_count, max_depth

def compute_effective_branching_factor(N, D, tol=1e-6):
    """
    Given:
      - N: total number of nodes generated (from depth 0 to D), and
      - D: maximum depth reached,
    compute the effective branching factor b* defined by:
    
        sum_{d=0}^{D} (b*)^d = N
    
    For b* ≠ 1, the series sums to ((b*)^(D+1) - 1)/(b* - 1).
    We solve the equation numerically via binary search.
    """
    # Handle the trivial case.
    if N == D + 1:
        return 1.0

    def f(b):
        # Avoid division by zero.
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

if __name__ == '__main__':
    # Use the standard solved state as the starting state.
    start_state = (1, 2, 3,
                   4, 5, 6,
                   7, 8, 0)
    
    branching, node_count, max_depth = bfs_enumeration_with_full_moves(start_state)
    
    # Print the total number of visited (unique) states.
    print("Total visited states:", node_count)
    print("Maximum depth reached:", max_depth)
    
    print("\nAverage full branching factor per depth (not subtracting reverse moves):")
    for depth in sorted(branching.keys()):
        avg = sum(branching[depth]) / len(branching[depth])
        print(f"  Depth {depth:2}: avg full BF = {avg:.2f} (from {len(branching[depth])} nodes)")

    # Compute the overall effective branching factor b*
    b_eff = compute_effective_branching_factor(node_count, max_depth)
    print("\nOverall effective branching factor b* = {:.4f}".format(b_eff))
