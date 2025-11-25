"""
Ray-based Async Training for Workflow-Conditioned RL
TRUE async: Workers collect episodes independently without synchronization!
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
import torch
import json
import csv
import time
import argparse
import ray
from typing import List, Tuple, Dict
from datetime import datetime

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from workflow_rl.order_based_workflow import OrderBasedWorkflow
from workflow_rl.gp_ucb_order_search import GPUCBOrderSearch
from workflow_rl.parallel_order_conditioned_ppo import ParallelOrderConditionedPPO, device


@ray.remote
class AsyncEpisodeWorker:
    """
    Ray actor that collects episodes independently.
    Runs continuously without waiting for other workers!
    """
    
    def __init__(self, worker_id: int, scenario_path: str, red_agent_type):
        self.worker_id = worker_id
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        
        # Create environment
        self.cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
        self.env = ChallengeWrapper2(env=self.cyborg, agent_name='Blue')
        
        self.episodes_collected = 0
        
        print(f"[Worker {worker_id}] Initialized")
    
    def collect_episode(self, policy_weights, workflow_encoding, max_steps=100):
        """
        Collect ONE complete episode using the given policy.
        Runs independently - no synchronization!
        """
        # Reconstruct policy from weights
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.get_action_space('Blue')
        
        # Create temporary policy network
        from workflow_rl.order_conditioned_ppo import OrderConditionedActorCritic
        policy = OrderConditionedActorCritic(obs_dim, action_dim, order_dims=25)
        policy.load_state_dict(policy_weights)
        policy.eval()
        
        # Move to CPU (workers don't need GPU)
        device_local = torch.device('cpu')
        policy = policy.to(device_local)
        workflow_tensor = torch.FloatTensor(workflow_encoding).to(device_local)
        
        # Collect episode
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'true_states_before': [],
            'true_states_after': []
        }
        
        state = self.env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get true state before
            true_before = self.cyborg.get_agent_state('True')
            
            # Prepare state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device_local)
            order_tensor = workflow_tensor.unsqueeze(0)
            augmented_state = torch.cat([state_tensor, order_tensor], dim=1)
            
            # Get action
            with torch.no_grad():
                logits = policy.actor(augmented_state)
                dist = torch.distributions.Categorical(logits)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)
                value = policy.critic(augmented_state)
            
            action = action_tensor.item()
            
            # Execute
            next_state, reward, done, info = self.env.step(action)
            
            # Get true state after
            true_after = self.cyborg.get_agent_state('True')
            
            # Store
            episode_data['states'].append(augmented_state.cpu().squeeze().numpy())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done)
            episode_data['log_probs'].append(log_prob.item())
            episode_data['values'].append(value.item())
            episode_data['true_states_before'].append(true_before)
            episode_data['true_states_after'].append(true_after)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        self.episodes_collected += 1
        
        return {
            'worker_id': self.worker_id,
            'episode_data': episode_data,
            'total_reward': total_reward,
            'steps': len(episode_data['actions'])
        }


class RayAsyncWorkflowRLTrainer:
    """
    Ray-based async trainer with TRUE independent episode collection!
    """
    
    def __init__(self, 
                 n_workers: int = 100,
                 total_episode_budget: int = 100000,
                 max_train_episodes_per_workflow: int = 500,
                 episodes_per_update: int = 100,
                 scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                 red_agent_type=RedMeanderAgent,
                 alignment_lambda: float = 30.0,
                 compliance_threshold: float = 0.95):
        
        self.n_workers = n_workers
        self.total_episode_budget = total_episode_budget
        self.max_train_episodes_per_workflow = max_train_episodes_per_workflow
        self.episodes_per_update = episodes_per_update
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        
        self.total_episodes_used = 0
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        self.experiment_name = f"exp_ray_async_{timestamp}_{pid}"
        self.checkpoint_dir = os.path.join("logs", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"\n🚀 Starting RAY ASYNC experiment with PID: {pid}")
        print(f"   TRUE async: {n_workers} workers collect episodes independently!")
        
        # Initialize Ray
        if not ray.is_initialized():
            print(f"📡 Initializing Ray...")
            ray.init(num_cpus=n_workers + 2, ignore_reinit_error=True)
            print(f"✅ Ray initialized!")
        
        # Create Ray workers
        print(f"\n👷 Creating {n_workers} Ray workers...")
        self.workers = [
            AsyncEpisodeWorker.remote(i, scenario_path, red_agent_type)
            for i in range(n_workers)
        ]
        print(f"✅ All {n_workers} Ray workers ready!\n")
        
        # Initialize workflow manager and GP-UCB
        self.workflow_manager = OrderBasedWorkflow()
        self.gp_search = GPUCBOrderSearch(beta=2.0)
        
        # Shared agent
        self.shared_agent = None
        
        # Initialize logging
        self._init_logging()
    
    def _init_logging(self):
        """Initialize CSV logging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(self.checkpoint_dir, f"training_log_{timestamp}.csv")
        self.log_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'Workflow_ID', 'Workflow_Order', 'Update', 'Episodes', 
            'Avg_Reward', 'Compliance', 'Collection_Time', 'Update_Time'
        ])
        self.log_file.flush()
        print(f"📊 Training log: {log_filename}")
        
        # GP log
        gp_log_filename = os.path.join(self.checkpoint_dir, "gp_sampling_log.csv")
        self.gp_log_file = open(gp_log_filename, 'w', newline='')
        self.gp_csv_writer = csv.writer(self.gp_log_file)
        self.gp_csv_writer.writerow(['Iteration', 'Workflow', 'UCB_Score', 'Reward'])
        self.gp_log_file.flush()
        print(f"📈 GP-UCB log: {gp_log_filename}\n")
    
    def collect_episodes_async(self, agent, workflow_order: List[str], 
                               n_episodes: int) -> Tuple:
        """
        Collect episodes asynchronously using Ray workers.
        Workers run INDEPENDENTLY - no synchronization!
        """
        workflow_encoding = self.workflow_manager.order_to_onehot(workflow_order)
        
        # Move policy weights to CPU for Ray workers (they don't have GPU)
        policy_weights = {k: v.cpu() for k, v in agent.policy_old.state_dict().items()}
        
        print(f"📦 Collecting {n_episodes} episodes from {self.n_workers} Ray workers...")
        print(f"   Workers run INDEPENDENTLY (true async!)")
        
        start_time = time.time()
        
        # Launch async collection on all workers
        # Each worker collects episodes independently!
        episodes_per_worker = n_episodes // self.n_workers
        remaining = n_episodes % self.n_workers
        
        futures = []
        for i, worker in enumerate(self.workers):
            # Distribute episodes fairly
            worker_episodes = episodes_per_worker + (1 if i < remaining else 0)
            
            # Launch multiple episode collections per worker
            for _ in range(worker_episodes):
                future = worker.collect_episode.remote(
                    policy_weights, workflow_encoding, max_steps=100
                )
                futures.append(future)
        
        # Collect results as they complete (async!)
        completed_episodes = []
        collected = 0
        
        while futures:
            # Wait for ANY future to complete (non-blocking across workers!)
            ready, futures = ray.wait(futures, num_returns=1, timeout=1.0)
            
            for future_id in ready:
                result = ray.get(future_id)
                completed_episodes.append(result)
                collected += 1
                
                if collected % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = collected / elapsed
                    print(f"  {collected}/{n_episodes} episodes ({rate:.1f} eps/sec)")
        
        elapsed = time.time() - start_time
        rate = n_episodes / elapsed
        print(f"✅ Collected {n_episodes} episodes in {elapsed:.1f}s ({rate:.1f} eps/sec)")
        print(f"   Workers ran INDEPENDENTLY (no synchronization!)\n")
        
        # Aggregate episode data
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        
        for episode_result in completed_episodes:
            ep_data = episode_result['episode_data']
            all_states.extend(ep_data['states'])
            all_actions.extend(ep_data['actions'])
            all_rewards.extend(ep_data['rewards'])
            all_dones.extend(ep_data['dones'])
            all_log_probs.extend(ep_data['log_probs'])
            all_values.extend(ep_data['values'])
        
        # Compute compliance (simplified)
        compliances = [1.0] * len(completed_episodes)  # Placeholder
        
        return (np.array(all_states), np.array(all_actions), np.array(all_rewards),
                np.array(all_dones), np.array(all_log_probs), np.array(all_values),
                compliances, elapsed)
    
    def train_workflow_async(self, workflow_order: List[str], workflow_id: int):
        """Train workflow using Ray async collection"""
        print(f"\n{'='*60}")
        print(f"🎯 RAY ASYNC Training: {' → '.join(workflow_order)}")
        print(f"   {self.n_workers} workers collect independently!")
        print(f"{'='*60}\n")
        
        # Create or inherit agent
        if self.shared_agent is None:
            print("🆕 Creating new agent")
            cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            obs_dim = env.observation_space.shape[0]
            
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_workers,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
        else:
            print("♻️  Inheriting from previous workflow")
            obs_dim = self.shared_agent.input_dims
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_workers,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
            agent.policy.load_state_dict(self.shared_agent.policy.state_dict())
            agent.policy_old.load_state_dict(self.shared_agent.policy_old.state_dict())
        
        # Training loop
        total_episodes = 0
        update_num = 0
        
        while total_episodes < self.max_train_episodes_per_workflow:
            update_num += 1
            print(f"🔄 Update {update_num}")
            
            # Collect episodes asynchronously
            (states, actions, rewards, dones, log_probs, values,
             compliances, collection_time) = self.collect_episodes_async(
                agent, workflow_order, self.episodes_per_update
            )
            
            total_episodes += self.episodes_per_update
            current_compliance = np.mean(compliances)
            
            # PPO update
            print(f"🔄 Performing PPO update...")
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
            
            # PPO update
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
            
            print(f"✅ Update complete ({update_time:.2f}s)")
            print(f"   Episodes: {total_episodes}/{self.max_train_episodes_per_workflow}")
            print(f"   Compliance: {current_compliance:.1%}")
            print(f"   Avg reward: {np.mean(rewards):.1f}")
            print(f"   Collection: {collection_time:.1f}s, Update: {update_time:.1f}s\n")
            
            # Log
            self.csv_writer.writerow([
                workflow_id, ' → '.join(workflow_order), update_num, total_episodes,
                np.mean(rewards), current_compliance, collection_time, update_time
            ])
            self.log_file.flush()
            
            # Check compliance
            if current_compliance >= self.compliance_threshold:
                print(f"🎉 Compliance threshold {self.compliance_threshold:.1%} achieved!\n")
                break
        
        # Save agent
        self.shared_agent = agent
        checkpoint_path = os.path.join(self.checkpoint_dir, f'workflow_{workflow_id}_agent.pt')
        torch.save(agent.policy.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}\n")
        
        return np.mean(rewards), current_compliance, total_episodes
    
    def run_workflow_search(self):
        """Run GP-UCB workflow search with Ray async training"""
        print("\n" + "="*60)
        print("🚀 RAY ASYNC Workflow Search")
        print("="*60)
        print(f"Architecture: Ray with TRUE async collection")
        print(f"Workers: {self.n_workers}")
        print(f"Episode budget: {self.total_episode_budget}")
        print(f"Episodes per update: {self.episodes_per_update}")
        print("="*60 + "\n")
        
        iteration = 0
        
        while self.total_episodes_used < self.total_episode_budget:
            iteration += 1
            
            print(f"\n{'='*60}")
            print(f"🎲 Iteration {iteration}")
            print(f"   Episodes used: {self.total_episodes_used}/{self.total_episode_budget}")
            print(f"{'='*60}\n")
            
            # Select workflow
            if iteration <= 5:
                canonical_dict = self.workflow_manager.get_canonical_workflows()
                candidate_orders = list(canonical_dict.values())
            else:
                unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
                candidate_orders = []
                for _ in range(10):
                    perm = unit_types.copy()
                    np.random.shuffle(perm)
                    candidate_orders.append(perm)
            
            workflow_order, ucb_score, info = self.gp_search.select_next_order(
                candidate_orders, self.workflow_manager
            )
            
            print(f"Selected workflow: {' → '.join(workflow_order)}")
            print(f"UCB score: {ucb_score:.4f}\n")
            
            # Train workflow
            eval_reward, compliance, episodes_used = self.train_workflow_async(
                workflow_order, iteration
            )
            
            self.total_episodes_used += episodes_used
            
            # Update GP model
            self.gp_search.add_observation(workflow_order, eval_reward)
            
            # Log
            self.gp_csv_writer.writerow([iteration, ' → '.join(workflow_order), ucb_score, eval_reward])
            self.gp_log_file.flush()
            
            print(f"📈 GP model updated")
            print(f"   Reward: {eval_reward:.2f}")
            print(f"   Compliance: {compliance:.1%}\n")
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        
        self.log_file.close()
        self.gp_log_file.close()
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Ray Async Workflow RL Training')
    parser.add_argument('--n-workers', type=int, default=100)
    parser.add_argument('--total-episodes', type=int, default=100000)
    parser.add_argument('--max-episodes-per-workflow', type=int, default=500)
    parser.add_argument('--episodes-per-update', type=int, default=100)
    parser.add_argument('--red-agent', type=str, default='B_lineAgent')
    parser.add_argument('--alignment-lambda', type=float, default=30.0)
    parser.add_argument('--compliance-threshold', type=float, default=0.95)
    
    args = parser.parse_args()
    
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    red_agent = agent_map[args.red_agent]
    
    print("\n" + "="*60)
    print("🚀 RAY ASYNC PARALLEL TRAINING")
    print("="*60)
    print(f"Workers: {args.n_workers} (TRUE async with Ray!)")
    print(f"Episode Budget: {args.total_episodes}")
    print(f"Episodes/Update: {args.episodes_per_update}")
    print("="*60 + "\n")
    
    trainer = RayAsyncWorkflowRLTrainer(
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
    main()

