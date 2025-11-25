"""
Async Parallel Environment - Workers Collect Episodes Independently
Each worker runs full episodes without synchronization at each step.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Process, Queue, shared_memory
import multiprocessing as mp
import queue
import time
import pickle
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


class AsyncEpisodeData:
    """Container for a completed episode"""
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.true_states_before = []
        self.true_states_after = []
        self.episode_reward = 0
        self.episode_steps = 0


def async_worker_process(worker_id: int, episode_queue: Queue, control_queue: Queue,
                         scenario_path: str, red_agent_type, max_steps: int = 100):
    """
    Worker that collects FULL EPISODES independently.
    No synchronization with other workers at each step!
    """
    # Create environment
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    print(f"[Worker {worker_id}] Started and ready")
    
    while True:
        try:
            # Check for control commands (non-blocking)
            try:
                cmd = control_queue.get_nowait()
                if cmd == 'stop':
                    print(f"[Worker {worker_id}] Stopping")
                    break
            except queue.Empty:
                pass
            
            # Collect a full episode independently
            episode = AsyncEpisodeData(worker_id)
            
            state = env.reset()
            episode.states.append(state)
            
            for step in range(max_steps):
                # Get true state before action
                true_state_before = cyborg.get_agent_state('True')
                episode.true_states_before.append(true_state_before)
                
                # For now, use a placeholder action (will be replaced by agent)
                # We'll actually get actions from a shared policy
                action = np.random.randint(0, env.get_action_space('Blue'))
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Get true state after action
                true_state_after = cyborg.get_agent_state('True')
                episode.true_states_after.append(true_state_after)
                
                # Store transition
                episode.actions.append(action)
                episode.rewards.append(reward)
                episode.dones.append(done)
                episode.states.append(next_state)
                
                episode.episode_reward += reward
                episode.episode_steps += 1
                
                if done:
                    break
            
            # Episode complete! Send to main process
            episode_queue.put(episode)
            
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            break


class ParallelEnvAsync:
    """
    Async parallel environment where workers collect episodes independently.
    Main process collects completed episodes without step-level synchronization.
    """
    
    def __init__(self, n_envs: int = 100, scenario_path: str = None,
                 red_agent_type=RedMeanderAgent, max_steps: int = 100):
        """
        Initialize async parallel environments.
        
        Unlike synchronous version, workers run FULL EPISODES independently!
        """
        self.n_envs = n_envs
        self.scenario_path = scenario_path or '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
        self.red_agent_type = red_agent_type
        self.max_steps = max_steps
        
        # Queue for completed episodes (workers push episodes here)
        self.episode_queue = Queue(maxsize=n_envs * 2)
        
        # Control queues for each worker
        self.control_queues = [Queue() for _ in range(n_envs)]
        
        # Get observation space
        cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': red_agent_type})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        self.observation_shape = env.observation_space.shape
        self.action_space = env.get_action_space('Blue')
        
        # Start worker processes
        self.workers = []
        print(f"\n🚀 Starting {n_envs} async workers (collecting episodes independently)...")
        
        for i in range(n_envs):
            p = Process(
                target=async_worker_process,
                args=(i, self.episode_queue, self.control_queues[i],
                      self.scenario_path, red_agent_type, max_steps),
                daemon=True
            )
            p.start()
            self.workers.append(p)
            
            # Stagger for large counts
            if (i + 1) % 50 == 0:
                print(f"  Started {i+1}/{n_envs} workers...")
                time.sleep(0.1)
        
        time.sleep(0.5)  # Let workers initialize
        print(f"✅ All {n_envs} async workers ready!\n")
    
    def collect_episodes(self, n_episodes: int, timeout: float = 300) -> List[AsyncEpisodeData]:
        """
        Collect n complete episodes from workers.
        Workers run independently - no synchronization at each step!
        
        Args:
            n_episodes: Number of episodes to collect
            timeout: Maximum time to wait for episodes
            
        Returns:
            List of completed episodes
        """
        episodes = []
        start_time = time.time()
        
        print(f"📦 Collecting {n_episodes} episodes from {self.n_envs} async workers...")
        
        while len(episodes) < n_episodes:
            try:
                # Non-blocking check with timeout
                episode = self.episode_queue.get(timeout=0.1)
                episodes.append(episode)
                
                if len(episodes) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(episodes) / elapsed
                    print(f"  Collected {len(episodes)}/{n_episodes} episodes ({rate:.1f} eps/sec)")
                
            except queue.Empty:
                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"⚠️  Timeout! Only collected {len(episodes)}/{n_episodes} episodes")
                    break
                continue
        
        elapsed = time.time() - start_time
        rate = len(episodes) / elapsed
        print(f"✅ Collected {len(episodes)} episodes in {elapsed:.1f}s ({rate:.1f} eps/sec)\n")
        
        return episodes
    
    def close(self):
        """Stop all workers"""
        print("Stopping async workers...")
        for q in self.control_queues:
            q.put('stop')
        
        for p in self.workers:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
        
        print("All workers stopped")


class AsyncEpisodeCollectorWithPolicy:
    """
    Enhanced async collector that uses a shared policy for action selection.
    Workers still collect episodes independently, but use the current policy.
    """
    
    def __init__(self, n_envs: int, scenario_path: str, red_agent_type,
                 max_steps: int = 100):
        self.n_envs = n_envs
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.max_steps = max_steps
        
        # Shared memory for policy parameters (to be implemented)
        # For now, we'll use simpler approach with episode queue
        
        # Get dimensions
        cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        self.observation_shape = env.observation_space.shape
        self.action_space = env.get_action_space('Blue')
    
    def collect_with_policy(self, agent, workflow_encoding: np.ndarray,
                           n_episodes: int) -> Tuple[List, List, List, List, List, List]:
        """
        Collect episodes using current policy.
        Each environment runs independently until episode completion.
        
        Returns:
            (states, actions, rewards, dones, log_probs, values)
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        all_true_states_before = []
        all_true_states_after = []
        
        # Create temporary environments
        envs = []
        cyborgs = []
        for _ in range(self.n_envs):
            cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            envs.append(env)
            cyborgs.append(cyborg)
        
        episodes_collected = 0
        env_episodes = [0] * self.n_envs  # Episodes per environment
        
        # Each environment collects episodes independently
        while episodes_collected < n_episodes:
            for env_id in range(self.n_envs):
                if episodes_collected >= n_episodes:
                    break
                
                # Run ONE full episode for this environment
                env = envs[env_id]
                cyborg = cyborgs[env_id]
                
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_dones = []
                episode_log_probs = []
                episode_values = []
                episode_true_before = []
                episode_true_after = []
                
                state = env.reset()
                
                for step in range(self.max_steps):
                    # Get true state before
                    true_before = cyborg.get_agent_state('True')
                    episode_true_before.append(true_before)
                    
                    # Get action from policy
                    state_tensor = agent._prepare_state(state, workflow_encoding)
                    action, log_prob, value = agent.get_action_single(state_tensor)
                    
                    # Execute
                    next_state, reward, done, info = env.step(action)
                    
                    # Get true state after
                    true_after = cyborg.get_agent_state('True')
                    episode_true_after.append(true_after)
                    
                    # Store
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    episode_dones.append(done)
                    episode_log_probs.append(log_prob)
                    episode_values.append(value)
                    
                    state = next_state
                    
                    if done:
                        break
                
                # Episode complete for this environment
                all_states.extend(episode_states)
                all_actions.extend(episode_actions)
                all_rewards.extend(episode_rewards)
                all_dones.extend(episode_dones)
                all_log_probs.extend(episode_log_probs)
                all_values.extend(episode_values)
                all_true_states_before.extend(episode_true_before)
                all_true_states_after.extend(episode_true_after)
                
                episodes_collected += 1
                env_episodes[env_id] += 1
        
        return (all_states, all_actions, all_rewards, all_dones, 
                all_log_probs, all_values, all_true_states_before, all_true_states_after)

