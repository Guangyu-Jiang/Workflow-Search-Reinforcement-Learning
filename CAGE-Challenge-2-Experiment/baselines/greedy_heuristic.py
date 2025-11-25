"""
Greedy Heuristic Baseline
Always fixes the highest-priority currently-compromised host according to workflow order
Should achieve ~100% compliance by design
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


# Action ID to host type mapping
ACTION_TO_HOST_TYPE = {
    # Remove actions
    15: 'defender', 16: 'enterprise', 17: 'enterprise', 18: 'enterprise',
    19: 'op_host', 20: 'op_host', 21: 'op_host', 22: 'op_server',
    23: 'user', 24: 'user', 25: 'user', 26: 'user', 27: 'user',
    # Restore actions
    132: 'defender', 133: 'enterprise', 134: 'enterprise', 135: 'enterprise',
    136: 'op_host', 137: 'op_host', 138: 'op_host', 139: 'op_server',
    140: 'user', 141: 'user', 142: 'user', 143: 'user', 144: 'user'
}

# Reverse mapping: host type to restore actions
HOST_TYPE_TO_RESTORE = {
    'defender': [132],
    'enterprise': [133, 134, 135],
    'op_host': [136, 137, 138],
    'op_server': [139],
    'user': [140, 141, 142, 143, 144]
}


def get_compromised_types(true_state):
    """Find which unit types are currently compromised"""
    compromised_types = set()
    
    for hostname, host_info in true_state.items():
        if hostname == 'success':
            continue
        
        # Check for Red agent sessions
        is_compromised = False
        if 'Sessions' in host_info:
            for session in host_info['Sessions']:
                if session.get('Agent') == 'Red':
                    is_compromised = True
                    break
        
        if is_compromised:
            # Determine unit type
            hostname_lower = hostname.lower()
            for unit_type in ['defender', 'enterprise', 'op_server', 'op_host', 'user']:
                if unit_type in hostname_lower or unit_type.replace('_', '') in hostname_lower:
                    compromised_types.add(unit_type)
                    break
    
    return compromised_types


def greedy_action_selection(true_state, workflow_order):
    """
    Greedy heuristic: Always fix highest-priority compromised host
    Returns the restore action for highest priority compromised type
    """
    compromised_types = get_compromised_types(true_state)
    
    if not compromised_types:
        # No compromised hosts - return Sleep action
        return 0
    
    # Find highest priority compromised type according to workflow
    highest_priority = None
    for unit_type in workflow_order:
        if unit_type in compromised_types:
            highest_priority = unit_type
            break
    
    if highest_priority is None:
        return 0  # Sleep
    
    # Select a restore action for this type
    possible_actions = HOST_TYPE_TO_RESTORE.get(highest_priority, [])
    if possible_actions:
        return np.random.choice(possible_actions)
    else:
        return 0


def run_greedy_heuristic(workflow_order,
                        n_episodes: int = 1000,
                        max_steps: int = 100,
                        red_agent_type=B_lineAgent,
                        scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'):
    """
    Run greedy heuristic baseline
    """
    
    workflow_str = ' → '.join(workflow_order)
    
    print("\n" + "="*60)
    print("GREEDY HEURISTIC BASELINE")
    print("="*60)
    print(f"Workflow: {workflow_str}")
    print(f"Episodes: {n_episodes}")
    print(f"Red Agent: {red_agent_type.__name__}")
    print(f"Policy: Always fix highest-priority compromised host")
    print("="*60 + "\n")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/greedy_heuristic_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create environment
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Logging
    log_file = open(os.path.join(exp_dir, 'training_log.csv'), 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['Episode', 'Reward', 'Steps', 'Compliance'])
    
    print(f"Experiment directory: {exp_dir}\n")
    
    # Run episodes
    episode_rewards = []
    episode_compliances = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        # Track compliance
        total_fix_actions = 0
        compliant_fix_actions = 0
        
        for step in range(max_steps):
            # Get true state
            true_state = cyborg.get_agent_state('True')
            
            # Greedy action selection
            action = greedy_action_selection(true_state, workflow_order)
            
            # Track compliance
            if action in ACTION_TO_HOST_TYPE:
                total_fix_actions += 1
                target_type = ACTION_TO_HOST_TYPE[action]
                
                compromised_types = get_compromised_types(true_state)
                if compromised_types:
                    highest_priority = None
                    for unit_type in workflow_order:
                        if unit_type in compromised_types:
                            highest_priority = unit_type
                            break
                    
                    if target_type == highest_priority:
                        compliant_fix_actions += 1
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Calculate compliance
        compliance = compliant_fix_actions / total_fix_actions if total_fix_actions > 0 else 1.0
        
        episode_rewards.append(episode_reward)
        episode_compliances.append(compliance)
        csv_writer.writerow([episode + 1, f"{episode_reward:.2f}", step + 1, f"{compliance:.4f}"])
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_compliance = np.mean(episode_compliances[-100:])
            print(f"Episode {episode + 1}/{n_episodes}: Reward={avg_reward:.2f}, Compliance={avg_compliance:.1%}")
    
    log_file.close()
    
    # Final statistics
    print("\n" + "="*60)
    print("✅ Greedy Heuristic Complete!")
    print(f"   Episodes: {n_episodes}")
    print(f"   Workflow: {workflow_str}")
    print(f"   Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Mean Compliance: {np.mean(episode_compliances):.1%}")
    print(f"   Std Reward: {np.std(episode_rewards):.2f}")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Greedy Heuristic Baseline')
    parser.add_argument('--n-episodes', type=int, default=1000)
    parser.add_argument('--workflow', type=str, default='defender,enterprise,op_server,op_host,user',
                       help='Comma-separated priority order')
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'])
    
    args = parser.parse_args()
    
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    
    workflow_order = args.workflow.split(',')
    
    run_greedy_heuristic(
        workflow_order=workflow_order,
        n_episodes=args.n_episodes,
        red_agent_type=agent_map[args.red_agent]
    )


if __name__ == "__main__":
    main()

