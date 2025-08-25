import chess
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np

mpl.rcParams['font.family'] = 'Palatino'
mpl.rcParams.update({
    'font.size': 26,         # Base font size
    'axes.titlesize': 26,    # Subplot title size
    'axes.labelsize': 26,    # Axis label size
    'xtick.labelsize': 20,   # x-tick label size
    'ytick.labelsize': 21,   # y-tick label size
    'legend.fontsize': 26    # Legend font size
})

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
    # Pad with zeros if the game ends before max_depth
    while len(branch_factors) < max_depth + 1:
        branch_factors.append(0)
    return branch_factors

def create_histogram_figure(
    depth_branch_factors,
    start_depth,
    end_depth,
    fig_title,
    max_cols=5,
    show_axis_labels_on_first=False
):
    """
    Plots sub-histograms for depths in [start_depth, end_depth)
    as a single figure with up to `max_cols` columns.
    We'll arrange subplots so that each figure can have multiple rows.
    """
    n = end_depth - start_depth
    if n <= 0:
        return  # nothing to plot

    rows = math.ceil(n / max_cols)
    fig, axs = plt.subplots(rows, max_cols, figsize=(max_cols * 4, rows * 4))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, depth in enumerate(range(start_depth, end_depth)):
        values = depth_branch_factors[depth]
        if not values:
            continue
        
        min_val = min(values)
        max_val = max(values)

        # If all values are the same, use a narrower bin of width 0.3
        if min_val == max_val:
            center = min_val
            bins = [center - 0.15, center + 0.15]
        else:
            # Create bin edges so each bar is centered on an integer
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

        # Use a lavender/purple color
        axs[i].hist(values, bins=bins, edgecolor='black', align='mid', color='purple')
        axs[i].set_title(f"Depth {depth}")

        # By default, remove x-axis and y-axis labels
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")

        # If this is the top-left subplot and we want axis labels there:
        if i == 0 and show_axis_labels_on_first:
            axs[i].set_xlabel("Legal Moves")
            axs[i].set_ylabel("Frequency")

        # Limit the number of x-ticks to 5
        max_ticks = 5
        range_val = max_val - min_val
        if range_val <= max_ticks:
            x_ticks = np.arange(min_val, max_val + 1)
        else:
            step = math.ceil(range_val / max_ticks)
            x_ticks = np.arange(min_val, max_val + 1, step)

        axs[i].set_xticks(x_ticks)
        axs[i].tick_params(axis='x', labelrotation=45)

        # Keep x-limits around the bins
        if min_val == max_val:
            axs[i].set_xlim(min_val - 0.5, min_val + 0.5)
        else:
            axs[i].set_xlim(min_val - 0.5, max_val + 0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    num_simulations = 10000
    max_depth = int(input("Enter the maximum depth (should be at least 140): "))

    depth_branch_factors = [[] for _ in range(max_depth + 1)]
    
    # Collect branching factor data from random games
    for _ in range(num_simulations):
        counts = simulate_random_game(max_depth)
        for d, count in enumerate(counts):
            depth_branch_factors[d].append(count)

    # Figure 1: Depths 1 to 10
    create_histogram_figure(
        depth_branch_factors,
        start_depth=1,
        end_depth=11,  # 11 is exclusive, so depths 1..10
        fig_title="Depth 1 to Depth 10 B.F. Distributions",
        max_cols=5,
        show_axis_labels_on_first=True  # show axis labels only on the first subplot
    )

    # Figure 2: Depths 131 to 140
    create_histogram_figure(
        depth_branch_factors,
        start_depth=131,
        end_depth=141,  # 141 is exclusive, so depths 131..140
        fig_title="Depth 131 to Depth 140 B.F. Distributions",
        max_cols=5,
        show_axis_labels_on_first=False
    )

if __name__ == '__main__':
    main()
