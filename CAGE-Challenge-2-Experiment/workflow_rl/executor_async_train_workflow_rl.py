"""
ProcessPoolExecutor-based Async Training
TRUE async using Python's built-in concurrent.futures - no external dependencies!
"""

import os
import sys
import warnings
import contextlib
import io

os.environ.setdefault('GYM_DISABLE_WARNINGS', '1')
os.environ.setdefault('GYM_LOG_LEVEL', 'ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
import torch
import json
import csv
import time
import argparse
from typing import List, Tuple, Dict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import gym
    gym.logger.set_level(gym.logger.ERROR)
except Exception:
    gym = None

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from workflow_rl.order_based_workflow import OrderBasedWorkflow
from workflow_rl.gp_ucb_order_search import GPUCBOrderSearch


def collect_single_episode(worker_id: int, scenario_path: str, red_agent_type,
                           policy_weights_cpu: Dict, workflow_encoding: np.ndarray,
                           workflow_order: List[str], alignment_lambda: float = 30.0,
                           compliant_bonus_scale: float = 0.0,
                           violation_penalty_scale: float = 0.0,
                           max_steps: int = 100):
    """
    Worker function that collects ONE complete episode.
    Runs in a separate process - completely independent!
    
    This function is submitted to ProcessPoolExecutor.
    """
    # Silence gym warnings inside worker processes as well
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import gym as _gym
        _gym.logger.set_level(_gym.logger.ERROR)
    except Exception:
        pass

    # Create environment in worker process
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Action ID to host type mapping (from CAGE2 action space)
    action_to_host_type = {
        # Remove actions
        15: 'defender', 16: 'enterprise', 17: 'enterprise', 18: 'enterprise',
        19: 'op_host', 20: 'op_host', 21: 'op_host', 22: 'op_server',
        23: 'user', 24: 'user', 25: 'user', 26: 'user', 27: 'user',
        # Restore actions  
        132: 'defender', 133: 'enterprise', 134: 'enterprise', 135: 'enterprise',
        136: 'op_host', 137: 'op_host', 138: 'op_host', 139: 'op_server',
        140: 'user', 141: 'user', 142: 'user', 143: 'user', 144: 'user'
    }
    
    # Reconstruct policy from weights (on CPU)
    import torch
    from workflow_rl.order_conditioned_ppo import OrderConditionedActorCritic
    
    obs_dim = env.observation_space.shape[0]
    if obs_dim != 52:
        raise RuntimeError(f"Unexpected observation dimension ({obs_dim}). Expected 52 to match baseline PPO input.")
    action_space = env.get_action_space('Blue')
    if isinstance(action_space, int):
        action_dim = action_space
    else:
        action_dim = len(action_space)
    if action_dim != 145:
        raise RuntimeError(f"Unexpected action dimension ({action_dim}). Expected 145 (full action space).")
    
    device_local = torch.device('cpu')  # Workers use CPU
    policy = OrderConditionedActorCritic(obs_dim, action_dim, order_dims=25).to(device_local)
    policy.load_state_dict(policy_weights_cpu)
    policy.eval()
    
    workflow_tensor = torch.FloatTensor(workflow_encoding).to(device_local)
    
    # Collect episode
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    episode_log_probs = []
    episode_values = []
    
    # Compliance tracking
    total_fix_actions = 0
    compliant_fix_actions = 0
    state = env.reset()
    total_env_reward = 0.0
    total_alignment_reward = 0.0
    steps = 0
    last_alignment_score = 0.0
    
    for step in range(max_steps):
        # Get TRUE STATE to check what's actually compromised
        true_state = cyborg.get_agent_state('True')
        
        # Prepare state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device_local)
        order_tensor = workflow_tensor.unsqueeze(0)
        augmented_state = torch.cat([state_tensor, order_tensor], dim=1)
        
        # Get action from policy
        with torch.no_grad():
            logits = policy.actor(augmented_state)
            dist = torch.distributions.Categorical(logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            value = policy.critic(augmented_state)
        
        action = action_tensor.item()
        
        # Execute
        next_state, reward, done, info = env.step(action)

        alignment_reward = 0.0
        step_bonus = 0.0
        
        # Track compliance for fix actions - check ACTUAL environment state
        if action in action_to_host_type:
            total_fix_actions += 1
            target_type = action_to_host_type[action]
            
            # Find which unit types are CURRENTLY compromised (from true state)
            compromised_types = set()
            for hostname, host_info in true_state.items():
                if hostname == 'success':
                    continue
                
                # Check if this host is compromised by checking for Red agent sessions
                is_compromised = False
                if 'Sessions' in host_info:
                    for session in host_info['Sessions']:
                        if session.get('Agent') == 'Red':
                            is_compromised = True
                            break
                
                if is_compromised:
                    # Determine unit type from hostname
                    hostname_lower = hostname.lower()
                    for unit_type in ['defender', 'enterprise', 'op_server', 'op_host', 'user']:
                        if unit_type in hostname_lower or unit_type.replace('_', '') in hostname_lower:
                            compromised_types.add(unit_type)
                            break
            
            # Find HIGHEST PRIORITY compromised type according to workflow
            highest_priority_compromised = None
            for unit_type in workflow_order:
                if unit_type in compromised_types:
                    highest_priority_compromised = unit_type
                    break  # First one in workflow order = highest priority
            
            # Check if agent is fixing the highest priority compromised type
            if highest_priority_compromised is None:
                # No compromised hosts - neutral fix
                compliant_fix_actions += 1
            elif target_type == highest_priority_compromised:
                # ✅ Fixing the highest priority compromised type!
                compliant_fix_actions += 1
                step_bonus = alignment_lambda * compliant_bonus_scale
            else:
                # ❌ Fixing lower priority when higher priority exists
                step_bonus = -alignment_lambda * violation_penalty_scale

            if total_fix_actions > 0:
                compliance_rate = compliant_fix_actions / total_fix_actions
                # Simple linear scaling (back to original)
                current_alignment_score = alignment_lambda * compliance_rate
            else:
                current_alignment_score = 0.0

            alignment_reward = current_alignment_score - last_alignment_score
            last_alignment_score = current_alignment_score
            alignment_reward += step_bonus

        if done and total_fix_actions == 0:
            # Penalize episodes that never attempted a fix (keep it mild)
            alignment_reward += -alignment_lambda * 0.2

        total_alignment_reward += alignment_reward
        
        # Store
        episode_states.append(augmented_state.squeeze().numpy())
        episode_actions.append(action)
        episode_rewards.append(reward + alignment_reward)
        episode_dones.append(done)
        episode_log_probs.append(log_prob.item())
        episode_values.append(value.item())
        
        total_env_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    # Compute episode compliance (same as synchronous version)
    if total_fix_actions > 0:
        episode_compliance = compliant_fix_actions / total_fix_actions
    else:
        # No fix actions - return 0.5 (neutral) to match synchronous version
        episode_compliance = 0.5
    
    total_reward = total_env_reward + total_alignment_reward

    return {
        'worker_id': worker_id,
        'states': episode_states,
        'actions': episode_actions,
        'rewards': episode_rewards,
        'dones': episode_dones,
        'log_probs': episode_log_probs,
        'values': episode_values,
        'total_reward': total_reward,
        'total_env_reward': total_env_reward,
        'total_alignment_reward': total_alignment_reward,
        'steps': steps,
        'compliance': episode_compliance,
        'total_fix_actions': total_fix_actions,
        'compliant_fix_actions': compliant_fix_actions
    }


class ExecutorAsyncWorkflowRLTrainer:
    """
    Async trainer using ProcessPoolExecutor.
    Workers collect full episodes independently!
    """
    
    def __init__(self, 
                 n_workers: int = 25,
                 total_episode_budget: int = 100000,
                 max_train_episodes_per_workflow: int = 200,
                 episodes_per_update: int = 25,
                 scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                 red_agent_type=RedMeanderAgent,
                 alignment_lambda: float = 100.0,
                 compliance_threshold: float = 0.90,
                 compliant_bonus_scale: float = 0.0,
                 violation_penalty_scale: float = 0.0,
                 compliance_focus_weight: float = 75.0,
                 patience_updates: int = 200,
                 verbose_collection: bool = False):
        
        self.n_workers = n_workers
        self.total_episode_budget = total_episode_budget
        self.max_train_episodes_per_workflow = max_train_episodes_per_workflow
        self.episodes_per_update = episodes_per_update
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        self.compliant_bonus_scale = compliant_bonus_scale
        self.violation_penalty_scale = violation_penalty_scale
        self.compliance_focus_weight = compliance_focus_weight
        self.patience_updates = patience_updates
        self.verbose_collection = verbose_collection
        
        self.total_episodes_used = 0
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        self.experiment_name = f"exp_executor_async_{timestamp}_{pid}"
        self.checkpoint_dir = os.path.join("logs", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"\nStarting experiment with PID: {pid}")
        
        # Create ProcessPoolExecutor
        self.executor = ProcessPoolExecutor(max_workers=n_workers)
        
        # Initialize workflow manager and GP-UCB
        self.workflow_manager = OrderBasedWorkflow()
        self.gp_search = GPUCBOrderSearch(beta=2.0)
        
        # Store trained policies per workflow (key: workflow tuple)
        # Only inherit if training the SAME workflow again!
        self.workflow_policies = {}  # {workflow_tuple: agent}
        self.shared_agent = None
        
        # Initialize logging
        self._init_logging()
    
    def _init_logging(self):
        """Initialize CSV logging"""
        # Save config
        config_file = os.path.join(self.checkpoint_dir, 'experiment_config.json')
        print(f"Experiment config: {config_file}")
        
        # Persist experiment configuration
        config = {
            "n_workers": self.n_workers,
            "total_episode_budget": self.total_episode_budget,
            "max_train_episodes_per_workflow": self.max_train_episodes_per_workflow,
            "episodes_per_update": self.episodes_per_update,
            "scenario_path": self.scenario_path,
            "red_agent": self.red_agent_type.__name__,
            "alignment_lambda": self.alignment_lambda,
            "compliance_threshold": self.compliance_threshold,
            "compliant_bonus_scale": self.compliant_bonus_scale,
            "violation_penalty_scale": self.violation_penalty_scale,
            "compliance_focus_weight": self.compliance_focus_weight,
            "patience_updates": self.patience_updates
        }
        with open(config_file, 'w') as cfg:
            json.dump(config, cfg, indent=2)
        
        # Training log
        log_filename = os.path.join(self.checkpoint_dir, "training_log.csv")
        self.log_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'Workflow_ID', 'Workflow_Order', 'Update', 'Episodes', 'Total_Episodes_Sampled',
            'Env_Reward', 'Alignment_Bonus', 'Total_Reward', 'Compliance',
            'Avg_Fixes', 'Collection_Time', 'Update_Time'
        ])
        self.log_file.flush()
        print(f"Experiment directory: {self.checkpoint_dir}")
        print(f"Training log: {log_filename}")
        
        # GP log
        gp_log_filename = os.path.join(self.checkpoint_dir, "gp_sampling_log.csv")
        self.gp_log_file = open(gp_log_filename, 'w', newline='')
        self.gp_csv_writer = csv.writer(self.gp_log_file)
        self.gp_csv_writer.writerow(['Iteration', 'Workflow', 'UCB_Score', 'Reward', 'Final_Compliance', 'Episodes_Trained', 'GP_Updated'])
        self.gp_log_file.flush()
        print(f"GP-UCB sampling log: {gp_log_filename}")
    
    def _find_closest_trained_workflow(self, workflow_order: List[str]):
        """Return the closest previously trained workflow (by Kendall distance)"""
        if not self.workflow_policies:
            return None, None, None
        
        best_key = None
        best_agent = None
        best_distance = float('inf')
        
        for existing_key, existing_agent in self.workflow_policies.items():
            existing_order = list(existing_key)
            distance = self.workflow_manager.compute_kendall_distance(existing_order, workflow_order)
            if distance < best_distance:
                best_distance = distance
                best_key = existing_key
                best_agent = existing_agent
        
        return best_key, best_agent, best_distance
    
    def collect_episodes_async(self, agent, workflow_order: List[str], 
                               n_episodes: int) -> Tuple:
        """
        Collect episodes asynchronously using ProcessPoolExecutor.
        Workers run INDEPENDENTLY - TRUE async!
        """
        workflow_encoding = self.workflow_manager.order_to_onehot(workflow_order)
        
        # Move policy weights to CPU for workers
        policy_weights_cpu = {k: v.cpu() for k, v in agent.policy_old.state_dict().items()}
        
        if self.verbose_collection:
            print(f"📦 Submitting {n_episodes} episode collection tasks to {self.n_workers} workers...")
            print(f"   Workers run INDEPENDENTLY (ProcessPoolExecutor async!)")
        
        start_time = time.time()
        
        # Submit episode collection tasks
        futures = []
        for i in range(n_episodes):
            # Each task collects ONE episode
            # Distribute across workers using worker_id % n_workers
            future = self.executor.submit(
                collect_single_episode,
                worker_id=i,
                scenario_path=self.scenario_path,
                red_agent_type=self.red_agent_type,
                policy_weights_cpu=policy_weights_cpu,
                workflow_encoding=workflow_encoding,
                workflow_order=workflow_order,
                alignment_lambda=self.alignment_lambda,
                compliant_bonus_scale=self.compliant_bonus_scale,
                violation_penalty_scale=self.violation_penalty_scale,
                max_steps=100
            )
            futures.append(future)
        
        if self.verbose_collection:
            print(f"✅ {n_episodes} tasks submitted! Collecting as they complete...")
        
        # Collect results as they complete (async!)
        completed_episodes = []
        collected = 0
        
        for future in as_completed(futures):
            try:
                result = future.result()
                completed_episodes.append(result)
                collected += 1
                
                if self.verbose_collection and (collected % 10 == 0 or collected == n_episodes):
                    elapsed = time.time() - start_time
                    rate = collected / elapsed
                    print(f"  {collected}/{n_episodes} episodes ({rate:.1f} eps/sec)")
                    
            except Exception as e:
                print(f"⚠️  Episode collection failed: {e}")
                continue
        
        elapsed = time.time() - start_time
        rate = len(completed_episodes) / elapsed
        if self.verbose_collection:
            print(f"✅ Collected {len(completed_episodes)} episodes in {elapsed:.1f}s ({rate:.1f} eps/sec)")
            print(f"   Workers ran INDEPENDENTLY (true async!)\n")
        
        # Aggregate episode data
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        compliances = []
        fix_counts = []
        episode_env_rewards = []
        episode_alignment_bonuses = []
        episode_total_rewards = []
        
        for episode in completed_episodes:
            all_states.extend(episode['states'])
            all_actions.extend(episode['actions'])
            all_rewards.extend(episode['rewards'])
            all_dones.extend(episode['dones'])
            all_log_probs.extend(episode['log_probs'])
            all_values.extend(episode['values'])
            compliances.append(episode['compliance'])
            fix_counts.append(episode['total_fix_actions'])
            episode_env_rewards.append(episode['total_env_reward'])
            episode_alignment_bonuses.append(episode['total_alignment_reward'])
            episode_total_rewards.append(episode['total_reward'])
        
        return (np.array(all_states), np.array(all_actions), np.array(all_rewards),
                np.array(all_dones), np.array(all_log_probs), np.array(all_values),
                compliances, fix_counts, episode_env_rewards, episode_alignment_bonuses,
                episode_total_rewards, elapsed)
    
    def train_workflow_async(self, workflow_order: List[str], workflow_id: int):
        """Train workflow using ProcessPoolExecutor async collection"""
        print(f"\n{'='*60}")
        print(f"Training with workflow: {' → '.join(workflow_order)}")
        print(f"Goal: Train until compliance >= {self.compliance_threshold:.1%}")
        print(f"Using {self.n_workers} async workers (ProcessPoolExecutor)")
        print(f"{'='*60}")
        
        # Check if this specific workflow has been trained before
        workflow_key = tuple(workflow_order)
        
        # Get observation dimensions
        cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': self.red_agent_type})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        obs_dim = env.observation_space.shape[0]
        if obs_dim != 52:
            raise RuntimeError(f"Unexpected observation dimension ({obs_dim}). Expected 52 to match baseline PPO input.")
        
        from workflow_rl.parallel_order_conditioned_ppo import ParallelOrderConditionedPPO, device
        
        if workflow_key in self.workflow_policies:
            # This EXACT workflow was trained before - inherit its policy!
            print(f"  Inheriting policy from previous training of THIS workflow")
            print(f"  (Resuming from checkpoint for this specific workflow)")
            
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_workers,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                compliant_bonus_scale=self.compliant_bonus_scale,
                violation_penalty_scale=self.violation_penalty_scale,
                K_epochs=6,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
            # Load weights from THIS workflow's previous training
            agent.policy.load_state_dict(self.workflow_policies[workflow_key].policy.state_dict())
            agent.policy_old.load_state_dict(self.workflow_policies[workflow_key].policy_old.state_dict())
            
        else:
            # New workflow - train from scratch!
            print(f"  Creating new agent (new workflow - training from scratch)")
            
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_workers,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                compliant_bonus_scale=self.compliant_bonus_scale,
                violation_penalty_scale=self.violation_penalty_scale,
                K_epochs=6,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
            
            closest_key, closest_agent, closest_distance = self._find_closest_trained_workflow(workflow_order)
            if closest_agent is not None:
                closest_order_str = ' → '.join(list(closest_key))
                print(f"  Initializing from closest trained workflow: {closest_order_str} (Kendall distance {closest_distance:.3f})")
                agent.policy.load_state_dict(closest_agent.policy.state_dict())
                agent.policy_old.load_state_dict(closest_agent.policy_old.state_dict())
            elif self.shared_agent is not None:
                print("  No similar workflow trained; seeding from shared backbone policy")
                agent.policy.load_state_dict(self.shared_agent.policy.state_dict())
                agent.policy_old.load_state_dict(self.shared_agent.policy_old.state_dict())
        
        # Training loop
        total_episodes = 0
        update_num = 0
        all_sampling_times = []
        all_update_times = []
        
        avg_env_reward = 0.0
        avg_alignment_bonus = 0.0
        avg_total_reward = 0.0
        avg_compliance = 0.0
        avg_fixes = 0.0
        best_compliance = 0.0
        updates_since_improvement = 0

        while total_episodes < self.max_train_episodes_per_workflow:
            
            # Collect episodes asynchronously
            (states, actions, rewards, dones, log_probs, values,
             compliances, fix_counts, episode_env_rewards,
             episode_alignment_bonuses, episode_total_rewards,
             collection_time) = self.collect_episodes_async(
                agent, workflow_order, self.episodes_per_update
            )
            
            update_num += 1
            total_episodes += self.episodes_per_update
            
            # Calculate metrics per episode
            avg_env_reward = np.mean(episode_env_rewards) if episode_env_rewards else 0.0
            avg_alignment_bonus = np.mean(episode_alignment_bonuses) if episode_alignment_bonuses else 0.0
            avg_total_reward = np.mean(episode_total_rewards) if episode_total_rewards else avg_env_reward + avg_alignment_bonus
            avg_compliance = np.mean(compliances) if compliances else 0.0
            avg_fixes = np.mean(fix_counts) if fix_counts else 0.0

            if avg_compliance > best_compliance + 0.01:
                best_compliance = avg_compliance
                updates_since_improvement = 0
            else:
                updates_since_improvement += 1
            
            # PPO update
            update_start = time.time()
            
            # Compute returns
            returns = []
            discounted_reward = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + agent.gamma * discounted_reward
                returns.insert(0, discounted_reward)
            
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
            # Convert to tensors
            old_states = torch.FloatTensor(states).to(device)
            old_actions = torch.LongTensor(actions).to(device)
            old_logprobs = torch.FloatTensor(log_probs).to(device)
            
            # PPO update (back to fixed K_epochs)
            for epoch in range(agent.K_epochs):
                logits = agent.policy.actor(old_states)
                state_values = agent.policy.critic(old_states).squeeze()
                
                dist = torch.distributions.Categorical(logits)
                new_logprobs = dist.log_prob(old_actions)
                entropy = dist.entropy()
                
                ratios = torch.exp(new_logprobs - old_logprobs)
                advantages = returns - state_values.detach()
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages
                
                loss = -torch.min(surr1, surr2).mean() + 0.5 * agent.MseLoss(state_values, returns) - 0.01 * entropy.mean()
                
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
            
            agent.policy_old.load_state_dict(agent.policy.state_dict())
            
            update_time = time.time() - update_start
            all_sampling_times.append(collection_time)
            all_update_times.append(update_time)
            
            # Calculate average timing statistics
            avg_sampling_time = np.mean(all_sampling_times)
            avg_update_time = np.mean(all_update_times)
            update_ratio = avg_update_time / (avg_sampling_time + avg_update_time) * 100
            
            # Print progress in same format as synchronous version
            print(f"\n  Update {update_num}: Episodes: {total_episodes} total")
            print(f"    Env Reward/Episode: {avg_env_reward:.2f}")
            print(f"    Total Reward/Episode: {avg_total_reward:.2f}")
            print(f"    Alignment Bonus (episode-end): {avg_alignment_bonus:+.2f}")
            print(f"    Compliance: {avg_compliance:.2%}")
            print(f"    Avg Fixes/Episode: {avg_fixes:.1f}")
            # Timing information removed for concise logs
            
            # Calculate cumulative episodes (across all workflows)
            cumulative_episodes = self.total_episodes_used + total_episodes
            
            # Log - separate environment reward, alignment bonus, and total reward
            self.csv_writer.writerow([
                workflow_id, ' → '.join(workflow_order), update_num, total_episodes,
                cumulative_episodes,                # Total episodes sampled from environment
                f"{avg_env_reward:.2f}",           # Original environment reward
                f"{avg_alignment_bonus:.2f}",      # Compliance/alignment reward
                f"{avg_total_reward:.2f}",         # Customized reward (sum)
                f"{avg_compliance:.4f}",           # Compliance rate
                f"{avg_fixes:.1f}",                # Average fixes per episode
                f"{collection_time:.2f}",          # Collection time
                f"{update_time:.2f}"               # Update time
            ])
            self.log_file.flush()
            
            # Check compliance
            if avg_compliance >= self.compliance_threshold:
                print(f"\n  ✓ Compliance threshold achieved!")
                print(f"    Episodes trained: {total_episodes}")
                print(f"    Latest compliance: {avg_compliance:.2%}")
                break
            if avg_compliance < self.compliance_threshold and updates_since_improvement >= self.patience_updates:
                print(f"\n  ⚠️  Compliance plateau detected (no improvement for {self.patience_updates} updates).")
                print(f"    Episodes trained: {total_episodes}")
                print(f"    Best compliance observed: {best_compliance:.2%}")
                break
        
        # Save agent for THIS specific workflow
        self.workflow_policies[workflow_key] = agent
        checkpoint_path = os.path.join(self.checkpoint_dir, f'workflow_{workflow_id}_agent.pt')
        torch.save(agent.policy.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint for this workflow: {checkpoint_path}")
        print(f"   Policy stored for workflow: {' → '.join(workflow_order)}\n")
        self.shared_agent = agent
        
        # GP-UCB receives the raw environment reward so workflow search optimizes the task objective.
        eval_reward = avg_env_reward
        return eval_reward, avg_compliance, total_episodes
    
    def run_workflow_search(self):
        """Run GP-UCB workflow search with ProcessPoolExecutor async training"""
        print(f"\n{'='*60}")
        print(f"Compliance-Gated Workflow Search (Async)")
        print(f"{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Directory: {self.checkpoint_dir}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Red Agent: {self.red_agent_type.__name__}")
        print(f"  Async Workers: {self.n_workers}")
        print(f"  Episode Budget: {self.total_episode_budget}")
        print(f"  Compliance Threshold: {self.compliance_threshold:.1%}")
        print(f"  Alignment Lambda: {self.alignment_lambda}")
        print(f"  Architecture: ProcessPoolExecutor (TRUE async!)")
        print(f"\nTraining Strategy:")
        print(f"  1. Train with alignment rewards until compliance >= {self.compliance_threshold:.1%}")
        print(f"  2. Use customized (env + alignment) rewards for GP-UCB")
        print(f"  3. Continue until {self.total_episode_budget} episodes are used")
        print(f"  4. Workers collect episodes INDEPENDENTLY (no sync barriers!)")
        print(f"{'='*60}")
        
        iteration = 0
        
        while self.total_episodes_used < self.total_episode_budget:
            iteration += 1
            
            print(f"\n{'='*50}")
            print(f"Iteration {iteration}")
            print(f"Episode Budget: {self.total_episodes_used}/{self.total_episode_budget} used")
            print(f"{'='*50}")
            
            # Select workflow - evaluate ALL possible permutations
            # Generate all 120 possible permutations of the 5 unit types
            from itertools import permutations as iter_permutations
            
            unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
            all_permutations = list(iter_permutations(unit_types))
            
            # Convert to list of lists
            candidate_orders = [list(perm) for perm in all_permutations]
            
            print(f"  Evaluating UCB for ALL {len(candidate_orders)} possible workflows...")
            
            # GP-UCB evaluates all candidates and picks highest UCB
            workflow_order, ucb_score, info = self.gp_search.select_next_order(
                candidate_orders, self.workflow_manager
            )
            
            # Print GP-UCB selection details (match synchronous format)
            print("\n" + "-"*50)
            print("GP-UCB Selection Details:")
            print(f"  Selected: {' → '.join(workflow_order)}")
            print(f"  UCB Score: {ucb_score:.3f}")
            
            if 'selection_method' in info:
                print(f"  Method: {info['selection_method']}")
                print(f"  Reason: {info['reason']}")
            else:
                if 'mean' in info:
                    print(f"  Mean Reward: {info['mean']:.2f}")
                if 'std' in info:
                    print(f"  Uncertainty (std): {info['std']:.3f}")
            print("-"*50)
            
            # Train workflow
            eval_reward, compliance, episodes_used = self.train_workflow_async(
                workflow_order, iteration
            )
            
            self.total_episodes_used += episodes_used
            
            # Update GP model ONLY if compliance threshold met
            # This ensures we only compare workflows that achieved the required compliance
            if compliance >= self.compliance_threshold:
                self.gp_search.add_observation(workflow_order, eval_reward)
                gp_updated = True
                print(f"📈 GP model updated (compliance {compliance:.1%} >= {self.compliance_threshold:.1%})")
                print(f"   Reward: {eval_reward:.2f}")
                print(f"   Episodes used: {episodes_used}")
            else:
                gp_updated = False
                print(f"⚠️  GP model NOT updated (compliance {compliance:.1%} < {self.compliance_threshold:.1%})")
                print(f"   Workflow did not meet compliance threshold")
                print(f"   Reward: {eval_reward:.2f} (not used for GP)")
                print(f"   Episodes used: {episodes_used}")
            
            # Log with final compliance and episodes trained
            self.gp_csv_writer.writerow([
                iteration, 
                ' → '.join(workflow_order), 
                f"{ucb_score:.4f}",
                f"{eval_reward:.2f}",
                f"{compliance:.4f}",  # Final compliance rate
                episodes_used,  # Total episodes used for this workflow
                'Yes' if gp_updated else 'No'  # Whether GP was updated
            ])
            self.gp_log_file.flush()
            
            print()
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print(f"   Total workflows explored: {iteration}")
        print(f"   Unique workflows trained: {len(self.workflow_policies)}")
        print(f"   Workflows trained from scratch: {len(self.workflow_policies)}")
        print(f"   Workflow re-visits: {iteration - len(self.workflow_policies)}")
        print("="*60)
        
        self.log_file.close()
        self.gp_log_file.close()
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser(description='ProcessPoolExecutor Async Workflow RL Training')
    parser.add_argument('--n-workers', type=int, default=50)
    parser.add_argument('--total-episodes', type=int, default=100000)
    parser.add_argument('--max-episodes-per-workflow', type=int, default=10000)
    parser.add_argument('--episodes-per-update', type=int, default=50)
    parser.add_argument('--red-agent', type=str, default='B_lineAgent')
    parser.add_argument('--alignment-lambda', type=float, default=100.0)
    parser.add_argument('--compliance-threshold', type=float, default=0.50)
    
    args = parser.parse_args()
    
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    red_agent = agent_map[args.red_agent]
    
    print(f"\n{'='*60}")
    print(f"Configuration")
    print(f"{'='*60}")
    print(f"Red Agent: {args.red_agent} ({red_agent.__name__})")
    print(f"Async Workers: {args.n_workers}")
    print(f"Episode Budget: {args.total_episodes}")
    print(f"Max Episodes/Workflow: {args.max_episodes_per_workflow}")
    print(f"Alignment Lambda: {args.alignment_lambda}")
    print(f"Compliance Threshold: {args.compliance_threshold:.1%}")
    print(f"Architecture: ProcessPoolExecutor (Async)")
    print(f"Random Seed: 42")
    print(f"{'='*60}\n")
    
    trainer = ExecutorAsyncWorkflowRLTrainer(
        n_workers=args.n_workers,
        total_episode_budget=args.total_episodes,
        max_train_episodes_per_workflow=args.max_episodes_per_workflow,
        episodes_per_update=args.episodes_per_update,
        red_agent_type=red_agent,
        alignment_lambda=args.alignment_lambda,
        compliance_threshold=args.compliance_threshold
    )
    
    trainer.run_workflow_search()


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some systems
    mp.set_start_method('spawn', force=True)
    main()
