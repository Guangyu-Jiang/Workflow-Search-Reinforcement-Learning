#!/usr/bin/env python3
"""
Browse and compare experiments from the logs directory
"""

import os
import json
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path
import argparse

def list_experiments(logs_dir: str = "logs"):
    """List all experiments in the logs directory"""
    
    if not os.path.exists(logs_dir):
        print(f"No logs directory found at {logs_dir}")
        return []
    
    experiments = []
    for exp_dir in sorted(os.listdir(logs_dir)):
        exp_path = os.path.join(logs_dir, exp_dir)
        if os.path.isdir(exp_path):
            # Try to load experiment config
            config_file = os.path.join(exp_path, "experiment_config.json")
            summary_file = os.path.join(exp_path, "summary.txt")
            log_file = os.path.join(exp_path, "training_log.csv")
            
            exp_info = {
                'name': exp_dir,
                'path': exp_path,
                'has_config': os.path.exists(config_file),
                'has_summary': os.path.exists(summary_file),
                'has_log': os.path.exists(log_file)
            }
            
            # Load config if available
            if exp_info['has_config']:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    exp_info['timestamp'] = config.get('timestamp', 'Unknown')
                    exp_info['red_agent'] = config.get('environment', {}).get('red_agent_type', 'Unknown')
                    exp_info['n_workflows'] = config.get('training', {}).get('n_workflows', 0)
                    exp_info['threshold'] = config.get('training', {}).get('compliance_threshold', 0)
                    exp_info['lambda'] = config.get('rewards', {}).get('alignment_lambda', 0)
            
            experiments.append(exp_info)
    
    return experiments

def show_experiment_summary(exp_path: str):
    """Display detailed summary of a specific experiment"""
    print(f"\n{'='*60}")
    print(f"Experiment: {os.path.basename(exp_path)}")
    print(f"{'='*60}")
    
    # Load and display config
    config_file = os.path.join(exp_path, "experiment_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("\nConfiguration:")
        print(f"  Timestamp: {config.get('timestamp', 'Unknown')}")
        print(f"  Red Agent: {config['environment']['red_agent_type']}")
        print(f"  Parallel Envs: {config['environment']['n_envs']}")
        print(f"  Workflows: {config['training']['n_workflows']}")
        print(f"  Compliance Threshold: {config['training']['compliance_threshold']:.1%}")
        print(f"  Alignment Lambda: {config['rewards']['alignment_lambda']}")
    
    # Load and display summary
    summary_file = os.path.join(exp_path, "summary.txt")
    if os.path.exists(summary_file):
        print("\nSummary:")
        print("-" * 40)
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            # Find and print key results
            for line in lines:
                if "Success Rate:" in line or "Best Workflow" in line:
                    print(line.strip())
    
    # Analyze CSV log
    log_file = os.path.join(exp_path, "training_log.csv")
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        print("\nTraining Statistics:")
        print("-" * 40)
        
        # Get workflow completions
        completions = df[df['Type'] == 'workflow_complete']
        if len(completions) > 0:
            n_success = (completions['Success'] == 'Yes').sum()
            print(f"  Workflows Trained: {len(completions)}")
            print(f"  Successful: {n_success}")
            print(f"  Success Rate: {n_success/len(completions):.1%}")
            
            # Best performing workflow
            successful = completions[completions['Success'] == 'Yes']
            if len(successful) > 0:
                best = successful.loc[successful['Eval_Reward'].astype(float).idxmax()]
                print(f"\n  Best Workflow:")
                print(f"    {best['Workflow_Order']}")
                print(f"    Eval Reward: {float(best['Eval_Reward']):.2f}")
                print(f"    Compliance: {float(best['Compliance']):.2%}")
        
        # Episode statistics
        episodes = df[df['Type'] == 'episode']
        if len(episodes) > 0:
            print(f"\n  Total Training Episodes: {len(episodes)}")
            print(f"  Mean Env Reward: {episodes['Env_Reward'].astype(float).mean():.2f}")
            print(f"  Mean Compliance: {episodes['Compliance'].astype(float).mean():.2%}")

def compare_experiments(exp_paths: list):
    """Compare multiple experiments side by side"""
    print(f"\n{'='*80}")
    print("Experiment Comparison")
    print(f"{'='*80}")
    
    comparison_data = []
    
    for exp_path in exp_paths:
        exp_name = os.path.basename(exp_path)
        data = {'Experiment': exp_name}
        
        # Load config
        config_file = os.path.join(exp_path, "experiment_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                data['Red Agent'] = config['environment']['red_agent_type']
                data['Threshold'] = f"{config['training']['compliance_threshold']:.0%}"
                data['Lambda'] = config['rewards']['alignment_lambda']
        
        # Load results from CSV
        log_file = os.path.join(exp_path, "training_log.csv")
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            completions = df[df['Type'] == 'workflow_complete']
            
            if len(completions) > 0:
                n_success = (completions['Success'] == 'Yes').sum()
                data['Success Rate'] = f"{n_success/len(completions):.1%}"
                data['Workflows'] = len(completions)
                
                successful = completions[completions['Success'] == 'Yes']
                if len(successful) > 0:
                    data['Best Reward'] = f"{successful['Eval_Reward'].astype(float).max():.1f}"
                else:
                    data['Best Reward'] = "N/A"
        
        comparison_data.append(data)
    
    # Display as table
    if comparison_data:
        df_compare = pd.DataFrame(comparison_data)
        print("\n" + df_compare.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Browse and compare experiments')
    parser.add_argument('--logs-dir', default='logs',
                       help='Logs directory (default: logs)')
    parser.add_argument('--list', action='store_true',
                       help='List all experiments')
    parser.add_argument('--show', type=str,
                       help='Show detailed summary of specific experiment')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple experiments')
    parser.add_argument('--latest', action='store_true',
                       help='Show the latest experiment')
    
    args = parser.parse_args()
    
    if args.list:
        experiments = list_experiments(args.logs_dir)
        if experiments:
            print(f"\n{'='*80}")
            print(f"Available Experiments in {args.logs_dir}/")
            print(f"{'='*80}")
            print(f"{'Name':<25} {'Timestamp':<20} {'Red Agent':<15} {'Workflows':<10} {'Threshold':<10}")
            print("-" * 80)
            
            for exp in experiments:
                name = exp['name'][:24]
                timestamp = exp.get('timestamp', 'Unknown')[:19]
                red_agent = exp.get('red_agent', 'Unknown')[:14]
                n_workflows = str(exp.get('n_workflows', 'N/A'))
                threshold = f"{exp.get('threshold', 0):.0%}" if 'threshold' in exp else 'N/A'
                
                print(f"{name:<25} {timestamp:<20} {red_agent:<15} {n_workflows:<10} {threshold:<10}")
            
            print(f"\nTotal experiments: {len(experiments)}")
            print(f"\nUse --show <exp_name> to see details of a specific experiment")
            print(f"Use --compare <exp1> <exp2> ... to compare multiple experiments")
    
    elif args.show:
        exp_path = os.path.join(args.logs_dir, args.show)
        if os.path.exists(exp_path):
            show_experiment_summary(exp_path)
        else:
            print(f"Experiment '{args.show}' not found in {args.logs_dir}")
    
    elif args.compare:
        exp_paths = [os.path.join(args.logs_dir, exp) for exp in args.compare]
        valid_paths = [p for p in exp_paths if os.path.exists(p)]
        
        if valid_paths:
            compare_experiments(valid_paths)
        else:
            print("No valid experiments found to compare")
    
    elif args.latest:
        experiments = list_experiments(args.logs_dir)
        if experiments:
            # Sort by name (which includes timestamp)
            latest = sorted(experiments, key=lambda x: x['name'])[-1]
            show_experiment_summary(latest['path'])
        else:
            print("No experiments found")
    
    else:
        # Default: list experiments
        parser.print_help()
        print("\n\nExamples:")
        print("  # List all experiments")
        print("  python browse_experiments.py --list")
        print("\n  # Show details of specific experiment")
        print("  python browse_experiments.py --show exp_20241015_143022")
        print("\n  # Compare multiple experiments")
        print("  python browse_experiments.py --compare exp_20241015_143022 exp_20241015_150000")
        print("\n  # Show latest experiment")
        print("  python browse_experiments.py --latest")

if __name__ == "__main__":
    main()
