#!/usr/bin/env python3
"""
Visualization utilities for Workflow-Conditioned RL results
Creates plots for slides and analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os


def plot_workflow_comparison(results_file: str, output_dir: str = "plots"):
    """Plot comparison of different workflows"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        history = json.load(f)
    
    # Extract data
    workflows = []
    rewards = []
    compliances = []
    
    for result in history:
        workflow_str = ' → '.join(result['workflow'])
        workflows.append(workflow_str)
        rewards.append(result['eval_reward'])
        compliances.append(result['eval_compliance'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    ax1.bar(range(len(workflows)), rewards, color='steelblue')
    ax1.set_xlabel('Workflow Iteration')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Workflow Performance Comparison')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add workflow labels (rotated for readability)
    ax1.set_xticks(range(len(workflows)))
    ax1.set_xticklabels([f"W{i+1}" for i in range(len(workflows))])
    
    # Plot compliance rates
    ax2.bar(range(len(workflows)), compliances, color='green')
    ax2.set_xlabel('Workflow Iteration')
    ax2.set_ylabel('Compliance Rate')
    ax2.set_title('Workflow Compliance Rates')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'workflow_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create legend figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    legend_text = "Workflow Legend:\n"
    for i, workflow in enumerate(workflows[:10]):  # Show first 10
        legend_text += f"W{i+1}: {workflow}\n"
    
    ax.text(0.1, 0.9, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.savefig(os.path.join(output_dir, 'workflow_legend.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(results_file: str, output_dir: str = "plots"):
    """Plot learning curves over iterations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(results_file, 'r') as f:
        history = json.load(f)
    
    iterations = [r['iteration'] for r in history]
    rewards = [r['eval_reward'] for r in history]
    
    # Compute running maximum
    running_max = []
    current_max = float('-inf')
    for r in rewards:
        current_max = max(current_max, r)
        running_max.append(current_max)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rewards, 'o-', label='Evaluated Reward', markersize=8)
    plt.plot(iterations, running_max, 'r--', label='Best So Far', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Workflow Search Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_workflow_heatmap(results_file: str, output_dir: str = "plots"):
    """Create heatmap of workflow performance by type ordering"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(results_file, 'r') as f:
        history = json.load(f)
    
    # Create position matrix
    unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
    position_rewards = {unit: {i: [] for i in range(5)} for unit in unit_types}
    
    for result in history:
        workflow = result['workflow']
        reward = result['eval_reward']
        
        for pos, unit in enumerate(workflow):
            if unit in unit_types:
                position_rewards[unit][pos].append(reward)
    
    # Compute average rewards
    heatmap_data = np.zeros((5, 5))
    for i, unit in enumerate(unit_types):
        for pos in range(5):
            if position_rewards[unit][pos]:
                heatmap_data[i, pos] = np.mean(position_rewards[unit][pos])
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=['1st', '2nd', '3rd', '4th', '5th'],
                yticklabels=unit_types,
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': 'Average Reward'})
    
    plt.xlabel('Position in Workflow')
    plt.ylabel('Unit Type')
    plt.title('Average Reward by Unit Type Position')
    
    plt.savefig(os.path.join(output_dir, 'position_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_architecture_diagram(output_dir: str = "plots"):
    """Create architecture diagram for slides"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define components
    components = {
        'GP-UCB': {'pos': (0.2, 0.8), 'size': (0.25, 0.15), 'color': 'lightblue'},
        'Workflow': {'pos': (0.2, 0.5), 'size': (0.25, 0.1), 'color': 'lightgreen'},
        'PPO': {'pos': (0.2, 0.2), 'size': (0.25, 0.15), 'color': 'lightcoral'},
        'CAGE2': {'pos': (0.6, 0.2), 'size': (0.25, 0.15), 'color': 'lightyellow'},
        'Alignment': {'pos': (0.6, 0.5), 'size': (0.25, 0.15), 'color': 'lightgray'}
    }
    
    # Draw components
    for name, props in components.items():
        rect = plt.Rectangle(props['pos'], props['size'][0], props['size'][1],
                           facecolor=props['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(props['pos'][0] + props['size'][0]/2, 
               props['pos'][1] + props['size'][1]/2,
               name, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw arrows
    arrows = [
        # GP-UCB to Workflow
        {'start': (0.325, 0.8), 'end': (0.325, 0.6), 'label': 'Select'},
        # Workflow to PPO
        {'start': (0.325, 0.5), 'end': (0.325, 0.35), 'label': 'Condition'},
        # PPO to CAGE2
        {'start': (0.45, 0.275), 'end': (0.6, 0.275), 'label': 'Action'},
        # CAGE2 to Alignment
        {'start': (0.725, 0.35), 'end': (0.725, 0.5), 'label': 'State'},
        # Alignment to PPO
        {'start': (0.6, 0.575), 'end': (0.45, 0.275), 'label': 'Reward'},
        # CAGE2 back to GP-UCB
        {'start': (0.85, 0.275), 'end': (0.85, 0.85), 'label': ''},
        {'start': (0.85, 0.85), 'end': (0.45, 0.85), 'label': 'Performance'}
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if arrow['label']:
            mid_x = (arrow['start'][0] + arrow['end'][0]) / 2
            mid_y = (arrow['start'][1] + arrow['end'][1]) / 2
            ax.text(mid_x, mid_y, arrow['label'], ha='center', va='bottom', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Workflow-Conditioned RL Architecture', fontsize=16, pad=20)
    
    plt.savefig(os.path.join(output_dir, 'architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_example_plots():
    """Create example plots with synthetic data"""
    
    output_dir = "example_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic results
    workflows = [
        ['defender', 'op_server', 'enterprise', 'op_host', 'user'],
        ['user', 'enterprise', 'op_host', 'defender', 'op_server'],
        ['enterprise', 'defender', 'op_server', 'op_host', 'user'],
        ['op_server', 'defender', 'enterprise', 'user', 'op_host'],
        ['defender', 'enterprise', 'op_server', 'user', 'op_host']
    ]
    
    history = []
    for i, workflow in enumerate(workflows):
        # Simulate that critical-first workflows perform better
        if workflow[0] in ['defender', 'op_server']:
            base_reward = -50 + np.random.normal(0, 10)
        else:
            base_reward = -80 + np.random.normal(0, 15)
        
        history.append({
            'iteration': i,
            'workflow': workflow,
            'train_reward': base_reward - 10,
            'train_compliance': 0.7 + np.random.uniform(0, 0.2),
            'eval_reward': base_reward,
            'eval_compliance': 0.8 + np.random.uniform(0, 0.15)
        })
    
    # Save synthetic data
    with open(os.path.join(output_dir, 'synthetic_results.json'), 'w') as f:
        json.dump(history, f)
    
    # Create all plots
    plot_workflow_comparison(os.path.join(output_dir, 'synthetic_results.json'), output_dir)
    plot_learning_curves(os.path.join(output_dir, 'synthetic_results.json'), output_dir)
    plot_workflow_heatmap(os.path.join(output_dir, 'synthetic_results.json'), output_dir)
    plot_architecture_diagram(output_dir)
    
    print(f"Example plots created in {output_dir}/")


if __name__ == "__main__":
    # Create example visualizations
    create_example_plots()
