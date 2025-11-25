"""
Analyze Remove and Restore actions in CAGE2
"""

import numpy as np
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

# Create environment
scenario_path = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
cyborg = CybORG(scenario_path, 'sim', agents={'Red': RedMeanderAgent})
env = ChallengeWrapper2(env=cyborg, agent_name='Blue')

# Get action info
print("Analyzing CAGE2 Action Space")
print("="*60)
print(f"Total actions: {env.action_space.n}")

# Get the raw action list from the environment
raw_cyborg = cyborg.environment_controller
blue_action_space = raw_cyborg.agent_interfaces['Blue'].action_space

print("\nAction mapping from wrapper:")
# The wrapper has an internal mapping
if hasattr(env, 'action_list'):
    for i, action in enumerate(env.action_list[:20]):
        print(f"  Action {i}: {action}")
    print("  ...")

# Look for Remove and Restore actions
print("\nRemove and Restore actions:")
remove_actions = []
restore_actions = []

# Check action names directly
for i in range(env.action_space.n):
    obs = env.reset()
    # Take the action and see what happens
    action_result = env.step(i)
    
    # Get the last action from CybORG
    last_action = cyborg.get_last_action('Blue')
    if last_action:
        action_str = str(last_action)
        if 'Remove' in action_str:
            remove_actions.append((i, action_str))
        elif 'Restore' in action_str:
            restore_actions.append((i, action_str))

# Print findings
print(f"\nFound {len(remove_actions)} Remove actions:")
for idx, name in remove_actions[:5]:
    print(f"  Action {idx}: {name}")

print(f"\nFound {len(restore_actions)} Restore actions:")
for idx, name in restore_actions[:5]:
    print(f"  Action {idx}: {name}")

# Check action indices from the original train.py
print("\nAction indices from train.py (132-144):")
for i in range(132, 145):
    obs = env.reset()
    env.step(i)
    last_action = cyborg.get_last_action('Blue')
    print(f"  Action {i}: {last_action}")

print("\n" + "="*60)
print("Key findings:")
print("1. Remove actions: Remove malicious processes/files")
print("2. Restore actions: Restore hosts to clean state")
print("3. Both are 'fix' actions that can repair compromised hosts")
