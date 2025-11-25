#!/usr/bin/env python3
"""
Script to run fair comparison experiments between baseline PPO and workflow search PPO
Ensures matching conditions for accurate comparison
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import json

def run_baseline_full_action(red_agent='meander', episodes=10000, exp_name=None):
    """Run baseline PPO with full action space"""
    
    if exp_name is None:
        exp_name = f"baseline_full_{red_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    print(f"Running Baseline PPO with FULL Action Space")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Red Agent: {red_agent}")
    print(f"Episodes: {episodes}")
    print(f"Action Space: 145 (full)")
    print(f"{'='*60}\n")
    
    cmd = [
        "python", "train_no_action_reduction.py",
        "--red-agent", red_agent,
        "--episodes", str(episodes)
    ]
    
    log_file = f"logs/{exp_name}_output.log"
    os.makedirs("logs", exist_ok=True)
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"Process started with PID: {process.pid}")
        print(f"Log file: {log_file}")
        return process, log_file


def run_baseline_reduced_action(red_agent='bline', episodes=10000, exp_name=None):
    """Run original baseline PPO with reduced action space (22 actions)"""
    
    if exp_name is None:
        exp_name = f"baseline_reduced_{red_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    print(f"Running Baseline PPO with REDUCED Action Space")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Red Agent: {red_agent}")
    print(f"Episodes: {episodes}")
    print(f"Action Space: 22 (reduced)")
    print(f"{'='*60}\n")
    
    # Modify train.py to use correct red agent
    modify_cmd = f"""
import fileinput
import sys

for line in fileinput.input('train.py', inplace=True):
    if 'Red': B_lineAgent' in line and '{red_agent}' == 'meander':
        print("        'Red': RedMeanderAgent")
    elif 'Red': RedMeanderAgent' in line and '{red_agent}' == 'bline':
        print("        'Red': B_lineAgent")
    else:
        print(line, end='')
"""
    
    # Run the modification
    subprocess.run(["python", "-c", modify_cmd])
    
    cmd = ["python", "train.py"]
    
    log_file = f"logs/{exp_name}_output.log"
    os.makedirs("logs", exist_ok=True)
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"Process started with PID: {process.pid}")
        print(f"Log file: {log_file}")
        return process, log_file


def run_workflow_search(red_agent='meander', n_workflows=20, max_episodes=400, 
                       alignment_lambda=30.0, compliance_threshold=0.95, exp_name=None):
    """Run workflow search PPO"""
    
    if exp_name is None:
        exp_name = f"workflow_search_{red_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    print(f"Running Workflow Search PPO")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Red Agent: {red_agent}")
    print(f"Workflows: {n_workflows}")
    print(f"Max Episodes/Env: {max_episodes}")
    print(f"Action Space: 145 (full)")
    print(f"Parallel Envs: 25")
    print(f"Alignment Lambda: {alignment_lambda}")
    print(f"Compliance Threshold: {compliance_threshold}")
    print(f"{'='*60}\n")
    
    cmd = [
        "python", "workflow_rl/parallel_train_workflow_rl.py",
        "--red-agent", red_agent,
        "--n-workflows", str(n_workflows),
        "--max-episodes", str(max_episodes),
        "--alignment-lambda", str(alignment_lambda),
        "--compliance-threshold", str(compliance_threshold)
    ]
    
    log_file = f"logs/{exp_name}_output.log"
    os.makedirs("logs", exist_ok=True)
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONPATH': os.getcwd()})
        print(f"Process started with PID: {process.pid}")
        print(f"Log file: {log_file}")
        return process, log_file


def create_comparison_config(name, red_agent='meander'):
    """Create configuration for fair comparison experiments"""
    
    config = {
        'experiment_name': name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'red_agent': red_agent,
        'experiments': {
            'baseline_reduced': {
                'name': f'{name}_baseline_reduced',
                'action_space': 22,
                'episodes': 10000,
                'update_steps': 20000,
                'k_epochs': 6,
                'parallel_envs': 1,
                'status': 'pending'
            },
            'baseline_full': {
                'name': f'{name}_baseline_full',
                'action_space': 145,
                'episodes': 10000,
                'update_steps': 20000,
                'k_epochs': 6,
                'parallel_envs': 1,
                'status': 'pending'
            },
            'workflow_search': {
                'name': f'{name}_workflow_search',
                'action_space': 145,
                'workflows': 20,
                'max_episodes_per_env': 400,
                'update_steps': 100,
                'k_epochs': 4,
                'parallel_envs': 25,
                'alignment_lambda': 30.0,
                'compliance_threshold': 0.95,
                'status': 'pending'
            }
        },
        'fair_comparison_notes': [
            "Baseline reduced has 6.6x advantage (22 vs 145 actions)",
            "Baseline collects 8x more data before updates (20k vs 2.5k steps)",
            "Workflow search uses 25x parallel environments",
            "For fairest comparison, use baseline_full vs workflow_search"
        ]
    }
    
    config_file = f"logs/{name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    return config


def main():
    parser = argparse.ArgumentParser(description='Run fair comparison experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['baseline-reduced', 'baseline-full', 'workflow-search', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--red-agent', type=str, default='meander',
                       choices=['meander', 'bline'],
                       help='Red agent type')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name prefix')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Episodes for baseline training')
    parser.add_argument('--workflows', type=int, default=20,
                       help='Number of workflows to search')
    parser.add_argument('--max-episodes', type=int, default=400,
                       help='Max episodes per workflow')
    
    args = parser.parse_args()
    
    # Set up experiment name
    if args.name is None:
        args.name = f"fair_comparison_{args.red_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Activate conda environment
    print("Setting up environment...")
    os.system("conda activate CAGE2")
    os.environ['PYTHONPATH'] = os.getcwd()
    
    processes = []
    
    if args.experiment == 'all':
        # Create comparison config
        config = create_comparison_config(args.name, args.red_agent)
        
        print("\n" + "="*60)
        print("Running ALL Comparison Experiments")
        print("="*60)
        
        # Run all three experiments
        p1, log1 = run_baseline_reduced_action(args.red_agent, args.episodes, 
                                              f"{args.name}_baseline_reduced")
        processes.append((p1, log1, 'baseline_reduced'))
        
        p2, log2 = run_baseline_full_action(args.red_agent, args.episodes,
                                           f"{args.name}_baseline_full")
        processes.append((p2, log2, 'baseline_full'))
        
        p3, log3 = run_workflow_search(args.red_agent, args.workflows, args.max_episodes,
                                      exp_name=f"{args.name}_workflow_search")
        processes.append((p3, log3, 'workflow_search'))
        
        print("\n" + "="*60)
        print("All experiments started!")
        print("Monitor progress with:")
        for p, log, name in processes:
            print(f"  tail -f {log}  # {name}")
        print("="*60)
        
    elif args.experiment == 'baseline-reduced':
        p, log = run_baseline_reduced_action(args.red_agent, args.episodes, args.name)
        processes.append((p, log, 'baseline_reduced'))
        
    elif args.experiment == 'baseline-full':
        p, log = run_baseline_full_action(args.red_agent, args.episodes, args.name)
        processes.append((p, log, 'baseline_full'))
        
    elif args.experiment == 'workflow-search':
        p, log = run_workflow_search(args.red_agent, args.workflows, args.max_episodes,
                                    exp_name=args.name)
        processes.append((p, log, 'workflow_search'))
    
    # Wait for processes if running single experiment
    if len(processes) == 1:
        p, log, name = processes[0]
        print(f"\nWaiting for {name} to complete...")
        print(f"Monitor with: tail -f {log}")
        p.wait()
        print(f"\n{name} completed with return code: {p.returncode}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
