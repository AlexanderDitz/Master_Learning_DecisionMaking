"""
Vector Flow Visualization Utilities

This module provides utilities for creating 2D vector flow plots to visualize model dynamics,
parameter changes, or any other 2D vector field data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax=None, 
                       arrow_max_num=200, arrow_alpha=0.8, plot_n_decimal=1):
    """
    Plot 2D vector flow field showing how variables change over time.
    
    Args:
        x1: Array of x1 values (starting points)
        x1_change: Array of x1 changes (vector components in x direction)
        x2: Array of x2 values (starting points)
        x2_change: Array of x2 changes (vector components in y direction)
        color: Color for the arrows
        axis_range: Tuple of (min, max) for axis limits
        ax: Matplotlib axis object (if None, current axis is used)
        arrow_max_num: Maximum number of arrows to plot (for performance)
        arrow_alpha: Transparency of arrows
        plot_n_decimal: Number of decimal places for tick labels
    
    Returns:
        ax: The matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()
    
    # Convert to numpy arrays if needed
    x1 = np.array(x1)
    x1_change = np.array(x1_change)
    x2 = np.array(x2)
    x2_change = np.array(x2_change)
    
    # Subsample arrows if there are too many
    if len(x1) > arrow_max_num:
        idx = np.random.choice(len(x1), arrow_max_num, replace=False)
        x1, x1_change, x2, x2_change = x1[idx], x1_change[idx], x2[idx], x2_change[idx]
    
    # Plot vector field
    ax.quiver(x1, x2, x1_change, x2_change, color=color,
              angles='xy', scale_units='xy', scale=1, alpha=arrow_alpha, 
              width=0.004, headwidth=10, headlength=10)
    
    # Set axis properties
    axis_min, axis_max = axis_range
    
    # Handle symmetric vs asymmetric ranges
    if axis_min < 0 < axis_max:
        axis_abs_max = max(abs(axis_min), abs(axis_max))
        axis_min, axis_max = -axis_abs_max, axis_abs_max
        ticks = [axis_min, 0, axis_max]
        ticklabels = [np.round(axis_min, plot_n_decimal), 0, np.round(axis_max, plot_n_decimal)]
    else:
        ticks = [axis_min, axis_max]
        ticklabels = [np.round(axis_min, plot_n_decimal), np.round(axis_max, plot_n_decimal)]
    
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_aspect('equal')
    
    return ax


def create_vector_flow_grid(data_dict, colors=None, figsize=(15, 10), save_path=None):
    """
    Create a grid of vector flow plots for multiple datasets.
    
    Args:
        data_dict: Dictionary where keys are labels and values are tuples of 
                  (x1, x1_change, x2, x2_change, axis_range)
        colors: List of colors for each plot (defaults to standard colors)
        figsize: Figure size tuple
        save_path: Path to save the figure (optional)
    
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    n_plots = len(data_dict)
    
    # Determine grid layout
    if n_plots <= 2:
        rows, cols = 1, n_plots
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    else:
        rows = int(np.ceil(np.sqrt(n_plots)))
        cols = int(np.ceil(n_plots / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Default colors
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Plot each dataset
    for i, (label, data) in enumerate(data_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        x1, x1_change, x2, x2_change, axis_range = data
        color = colors[i % len(colors)]
        
        plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
    
    # Hide unused subplots
    for i in range(len(data_dict), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Vector flow grid saved to: {save_path}")
    
    return fig, axes


def generate_sample_dynamics(dynamics_type='spiral', n_points=500, noise_level=0.1):
    """
    Generate sample dynamics data for testing vector flow plots.
    
    Args:
        dynamics_type: Type of dynamics ('spiral', 'saddle', 'center', 'sink')
        n_points: Number of data points to generate
        noise_level: Amount of noise to add
    
    Returns:
        x1, x1_change, x2, x2_change, axis_range: Data for vector flow plot
    """
    # Generate initial positions
    x1 = np.random.uniform(-2, 2, n_points)
    x2 = np.random.uniform(-2, 2, n_points)
    
    if dynamics_type == 'spiral':
        # Spiral dynamics
        x1_change = -0.1 * x1 - 0.5 * x2 + noise_level * np.random.randn(n_points)
        x2_change = 0.5 * x1 - 0.1 * x2 + noise_level * np.random.randn(n_points)
    elif dynamics_type == 'saddle':
        # Saddle point dynamics
        x1_change = 0.5 * x1 + noise_level * np.random.randn(n_points)
        x2_change = -0.5 * x2 + noise_level * np.random.randn(n_points)
    elif dynamics_type == 'center':
        # Center dynamics (circular)
        x1_change = -x2 + noise_level * np.random.randn(n_points)
        x2_change = x1 + noise_level * np.random.randn(n_points)
    elif dynamics_type == 'sink':
        # Sink dynamics (converging)
        x1_change = -0.3 * x1 + noise_level * np.random.randn(n_points)
        x2_change = -0.3 * x2 + noise_level * np.random.randn(n_points)
    else:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")
    
    axis_range = (-2.5, 2.5)
    return x1, x1_change, x2, x2_change, axis_range


def demo_vector_flow():
    """
    Create a demonstration of different vector flow patterns in 2D.
    """
    print("Creating 2D vector flow demonstration...")
    
    # Generate different 2D dynamics
    dynamics_types_2d = ['spiral', 'saddle', 'center', 'sink']
    data_dict_2d = {}
    
    for dyn_type in dynamics_types_2d:
        x1, x1_change, x2, x2_change, axis_range = generate_sample_dynamics(dyn_type)
        data_dict_2d[dyn_type.capitalize()] = (x1, x1_change, x2, x2_change, axis_range)
    
    # Create the 2D plot
    fig_2d, axes_2d = create_vector_flow_grid(
        data_dict_2d, 
        figsize=(12, 8),
        save_path="vector_flow_2d_demo.png"
    )
    
    plt.suptitle('2D Vector Flow Patterns Demo', fontsize=16, fontweight='bold')
    plt.show()
    
    return fig_2d, axes_2d


if __name__ == "__main__":
    # Run demos
    print("Creating vector flow demonstrations...")
    demo_vector_flow()
    print("Demos complete!")
