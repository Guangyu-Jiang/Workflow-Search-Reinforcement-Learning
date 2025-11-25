"""
Fixed Priority Workflow Baselines
Tests multiple predefined workflow orders to compare with GP-UCB discovered workflows
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
import csv
from datetime import datetime
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

# Import greedy heuristic
from baselines.greedy_heuristic import run_greedy_heuristic


# Predefined workflow strategies
CANONICAL_WORKFLOWS = {
    'critical_first': ['defender', 'op_server', 'enterprise', 'op_host', 'user'],
    'enterprise_focus': ['enterprise', 'defender', 'op_server', 'op_host', 'user'],
    'user_priority': ['user', 'defender', 'enterprise', 'op_server', 'op_host'],
    'operational_focus': ['op_server', 'op_host', 'defender', 'enterprise', 'user'],
    'balanced': ['defender', 'enterprise', 'op_server', 'user', 'op_host'],
    'reverse': ['user', 'op_host', 'enterprise', 'op_server', 'defender']
}


def evaluate_fixed_workflows(n_episodes_per_workflow: int = 1000,
                             red_agent_type=B_lineAgent,
                             scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'):
    """
    Evaluate all canonical fixed-priority workflows
    """
    
    print("\n" + "="*60)
    print("FIXED PRIORITY WORKFLOWS BASELINE")
    print("="*60)
    print(f"Testing {len(CANONICAL_WORKFLOWS)} predefined workflows")
    print(f"Episodes per workflow: {n_episodes_per_workflow}")
    print(f"Red Agent: {red_agent_type.__name__}")
    print("="*60 + "\n")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/fixed_workflows_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Summary log
    summary_file = open(os.path.join(exp_dir, 'summary.csv'), 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['Workflow_Name', 'Workflow_Order', 'Mean_Reward', 'Std_Reward', 
                            'Mean_Compliance', 'Min_Reward', 'Max_Reward'])
    
    results = []
    
    for workflow_name, workflow_order in CANONICAL_WORKFLOWS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {workflow_name}")
        print(f"Order: {' → '.join(workflow_order)}")
        print(f"{'='*60}\n")
        
        # Run greedy heuristic with this workflow
        run_greedy_heuristic(
            workflow_order=workflow_order,
            n_episodes=n_episodes_per_workflow,
            red_agent_type=red_agent_type,
            scenario_path=scenario_path
        )
        
        # Load results
        log_file = f"logs/greedy_heuristic_*/training_log.csv"
        import glob
        latest_log = sorted(glob.glob(log_file))[-1]
        
        import pandas as pd
        df = pd.read_csv(latest_log)
        
        mean_reward = df['Reward'].mean()
        std_reward = df['Reward'].std()
        mean_compliance = df['Compliance'].mean()
        min_reward = df['Reward'].min()
        max_reward = df['Reward'].max()
        
        summary_writer.writerow([
            workflow_name,
            ' → '.join(workflow_order),
            f"{mean_reward:.2f}",
            f"{std_reward:.2f}",
            f"{mean_compliance:.4f}",
            f"{min_reward:.2f}",
            f"{max_reward:.2f}"
        ])
        
        results.append({
            'name': workflow_name,
            'order': workflow_order,
            'reward': mean_reward,
            'compliance': mean_compliance
        })
    
    summary_file.close()
    
    # Print comparison
    print("\n" + "="*60)
    print("FIXED WORKFLOW COMPARISON")
    print("="*60)
    print(f"\n{'Workflow':<20} {'Reward':>10} {'Compliance':>12}")
    print("-" * 45)
    
    # Sort by reward
    results.sort(key=lambda x: x['reward'], reverse=True)
    
    for r in results:
        print(f"{r['name']:<20} {r['reward']:>10.2f} {r['compliance']:>11.1%}")
    
    print("\n" + "="*60)
    print(f"Best workflow: {results[0]['name']}")
    print(f"  Order: {' → '.join(results[0]['order'])}")
    print(f"  Reward: {results[0]['reward']:.2f}")
    print(f"  Compliance: {results[0]['compliance']:.1%}")
    print("="*60)
    
    print(f"\nSummary saved to: {exp_dir}/summary.csv")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Fixed Priority Workflows')
    parser.add_argument('--n-episodes', type=int, default=1000,
                       help='Episodes per workflow')
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'])
    
    args = parser.parse_args()
    
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    
    evaluate_fixed_workflows(
        n_episodes_per_workflow=args.n_episodes,
        red_agent_type=agent_map[args.red_agent]
    )


if __name__ == "__main__":
    main()

