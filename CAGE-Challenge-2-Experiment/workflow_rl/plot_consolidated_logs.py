"""
Plot training metrics from consolidated CSV log file
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_consolidated_training(csv_file: str, output_dir: str = None):
    """
    Plot training metrics from consolidated CSV log
    
    Args:
        csv_file: Path to consolidated training log CSV
        output_dir: Directory to save plots (optional)
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get unique workflows
    workflows = df[df['Type'] == 'episode']['Workflow_ID'].unique()
    n_workflows = len(workflows)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Compliance progression for each workflow
    ax1 = plt.subplot(2, 3, 1)
    for wf_id in workflows[:10]:  # Show first 10 workflows
        wf_data = df[(df['Workflow_ID'] == wf_id) & (df['Type'] == 'episode')]
        if len(wf_data) > 0:
            # Group by total episodes across all envs
            episode_groups = wf_data.groupby('Episode').agg({
                'Compliance': 'mean'
            }).reset_index()
            ax1.plot(episode_groups['Episode'], episode_groups['Compliance'], 
                    label=f'WF {wf_id}', alpha=0.7)
    ax1.axhline(y=0.95, color='r', linestyle='--', label='Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Compliance Rate')
    ax1.set_title('Compliance Progression by Workflow')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Environment rewards over time
    ax2 = plt.subplot(2, 3, 2)
    for wf_id in workflows[:10]:
        wf_data = df[(df['Workflow_ID'] == wf_id) & (df['Type'] == 'episode')]
        if len(wf_data) > 0:
            episode_groups = wf_data.groupby('Episode').agg({
                'Env_Reward': 'mean'
            }).reset_index()
            ax2.plot(episode_groups['Episode'], episode_groups['Env_Reward'], 
                    label=f'WF {wf_id}', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Environment Reward')
    ax2.set_title('Environment Rewards by Workflow')
    ax2.grid(True, alpha=0.3)
    
    # 3. Success rate across workflows
    ax3 = plt.subplot(2, 3, 3)
    workflow_complete = df[df['Type'] == 'workflow_complete']
    if len(workflow_complete) > 0:
        success_data = workflow_complete.copy()
        success_data['Success_Binary'] = success_data['Success'].map({'Yes': 1, 'No': 0})
        success_rate = []
        for i in range(1, len(success_data) + 1):
            rate = success_data.iloc[:i]['Success_Binary'].mean()
            success_rate.append(rate)
        ax3.plot(range(1, len(success_rate) + 1), success_rate, 'b-', linewidth=2)
        ax3.fill_between(range(1, len(success_rate) + 1), success_rate, alpha=0.3)
    ax3.set_xlabel('Workflow Number')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Cumulative Success Rate')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Evaluation rewards for successful workflows
    ax4 = plt.subplot(2, 3, 4)
    successful = workflow_complete[workflow_complete['Success'] == 'Yes']
    if len(successful) > 0:
        eval_rewards = successful['Eval_Reward'].astype(float)
        ax4.bar(range(len(eval_rewards)), eval_rewards, color='green', alpha=0.7)
        ax4.axhline(y=eval_rewards.mean(), color='r', linestyle='--', 
                   label=f'Mean: {eval_rewards.mean():.2f}')
        ax4.set_xlabel('Successful Workflow Index')
        ax4.set_ylabel('Evaluation Reward')
        ax4.set_title('Evaluation Rewards (Successful Only)')
        ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Training episodes required
    ax5 = plt.subplot(2, 3, 5)
    if len(workflow_complete) > 0:
        episodes_trained = workflow_complete['Total_Episodes'].astype(float) / 25  # Per env
        colors = ['green' if s == 'Yes' else 'red' 
                 for s in workflow_complete['Success']]
        ax5.bar(range(len(episodes_trained)), episodes_trained, color=colors, alpha=0.7)
        ax5.axhline(y=100, color='r', linestyle='--', label='Max Episodes')
        ax5.set_xlabel('Workflow Number')
        ax5.set_ylabel('Episodes per Environment')
        ax5.set_title('Training Episodes Required')
        ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Workflow diversity (show top workflows)
    ax6 = plt.subplot(2, 3, 6)
    if len(workflow_complete) > 0:
        # Count frequency of each workflow order
        workflow_counts = workflow_complete['Workflow_Order'].value_counts().head(10)
        if len(workflow_counts) > 0:
            y_pos = np.arange(len(workflow_counts))
            ax6.barh(y_pos, workflow_counts.values, alpha=0.7)
            ax6.set_yticks(y_pos)
            # Shorten workflow names for display
            labels = [w.split(' → ')[0] + '...' + w.split(' → ')[-1] 
                     for w in workflow_counts.index]
            ax6.set_yticklabels(labels, fontsize=8)
            ax6.set_xlabel('Times Explored')
            ax6.set_title('Top 10 Most Explored Workflows')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Consolidated Training Analysis: {os.path.basename(csv_file)}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_path = os.path.join(output_dir, 'consolidated_training_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    
    if len(workflow_complete) > 0:
        n_successful = (workflow_complete['Success'] == 'Yes').sum()
        success_rate = n_successful / len(workflow_complete)
        
        print(f"Total Workflows Explored: {len(workflow_complete)}")
        print(f"Successful Workflows: {n_successful}")
        print(f"Success Rate: {success_rate:.1%}")
        
        if n_successful > 0:
            successful = workflow_complete[workflow_complete['Success'] == 'Yes']
            eval_rewards = successful['Eval_Reward'].astype(float)
            print(f"\nSuccessful Workflows Statistics:")
            print(f"  Best Eval Reward: {eval_rewards.max():.2f}")
            print(f"  Mean Eval Reward: {eval_rewards.mean():.2f}")
            print(f"  Std Eval Reward: {eval_rewards.std():.2f}")
            
            # Find best workflow
            best_idx = eval_rewards.idxmax()
            best_workflow = successful.loc[best_idx, 'Workflow_Order']
            print(f"\nBest Workflow: {best_workflow}")
            print(f"  Reward: {eval_rewards.max():.2f}")
            print(f"  Compliance: {successful.loc[best_idx, 'Compliance']}")
    
    # Episode statistics
    episode_data = df[df['Type'] == 'episode']
    if len(episode_data) > 0:
        print(f"\nEpisode Statistics:")
        print(f"  Total Episodes: {len(episode_data)}")
        print(f"  Mean Env Reward: {episode_data['Env_Reward'].astype(float).mean():.2f}")
        print(f"  Mean Compliance: {episode_data['Compliance'].astype(float).mean():.4f}")
        print(f"  Mean Fixes: {episode_data['Fixes'].astype(float).mean():.2f}")
    
    print("="*60)

def analyze_workflow_progression(csv_file: str, workflow_id: int):
    """
    Detailed analysis of a specific workflow's training progression
    
    Args:
        csv_file: Path to consolidated training log CSV
        workflow_id: ID of workflow to analyze
    """
    df = pd.read_csv(csv_file)
    
    # Filter for specific workflow
    wf_episodes = df[(df['Workflow_ID'] == workflow_id) & (df['Type'] == 'episode')]
    wf_summary = df[(df['Workflow_ID'] == workflow_id) & (df['Type'] == 'summary')]
    wf_complete = df[(df['Workflow_ID'] == workflow_id) & (df['Type'] == 'workflow_complete')]
    
    if len(wf_episodes) == 0:
        print(f"No data found for workflow {workflow_id}")
        return
    
    # Get workflow order
    workflow_order = wf_episodes.iloc[0]['Workflow_Order']
    print(f"\nWorkflow {workflow_id}: {workflow_order}")
    print("-" * 60)
    
    # Plot progression
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Compliance over episodes
    episode_groups = wf_episodes.groupby('Episode').agg({
        'Compliance': 'mean',
        'Env_Reward': 'mean',
        'Total_Reward': 'mean',
        'Fixes': 'mean'
    }).reset_index()
    
    axes[0, 0].plot(episode_groups['Episode'], episode_groups['Compliance'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='Threshold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Compliance Rate')
    axes[0, 0].set_title('Compliance Progression')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rewards over episodes
    axes[0, 1].plot(episode_groups['Episode'], episode_groups['Env_Reward'], 
                   'g-', label='Env Reward', linewidth=2)
    axes[0, 1].plot(episode_groups['Episode'], episode_groups['Total_Reward'], 
                   'b--', label='Total Reward', linewidth=1, alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Reward Progression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fixes over episodes
    axes[1, 0].plot(episode_groups['Episode'], episode_groups['Fixes'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Fixes')
    axes[1, 0].set_title('Fix Actions per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    if len(wf_complete) > 0:
        success = wf_complete.iloc[0]['Success']
        eval_reward = float(wf_complete.iloc[0]['Eval_Reward'])
        final_compliance = float(wf_complete.iloc[0]['Compliance'])
        total_episodes = int(wf_complete.iloc[0]['Total_Episodes'])
        
        stats_text = f"Workflow Complete Summary:\n\n"
        stats_text += f"Success: {success}\n"
        stats_text += f"Eval Reward: {eval_reward:.2f}\n"
        stats_text += f"Final Compliance: {final_compliance:.4f}\n"
        stats_text += f"Total Episodes: {total_episodes}\n"
        stats_text += f"Episodes per Env: {total_episodes/25:.1f}\n"
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                       verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.suptitle(f'Workflow {workflow_id} Analysis: {workflow_order[:50]}...', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_consolidated_logs.py <csv_file> [output_dir]")
        print("\nExample:")
        print("  python plot_consolidated_logs.py compliance_checkpoints/training_log_20241015_143022.csv")
        print("\nFor specific workflow analysis:")
        print("  python -c \"from plot_consolidated_logs import analyze_workflow_progression; "
              "analyze_workflow_progression('file.csv', workflow_id=0)\"")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_consolidated_training(csv_file, output_dir)
