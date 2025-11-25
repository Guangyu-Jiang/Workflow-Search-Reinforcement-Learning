"""
Parallel Environment Wrapper for CAGE2
Allows running multiple environments in parallel for more stable PPO training
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from multiprocessing import Process, Pipe
import multiprocessing as mp
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


def worker_process(remote, parent_remote, env_fn):
    """Worker process for running environment"""
    parent_remote.close()
    env, cyborg = env_fn()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
                
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
                
            elif cmd == 'get_true_state':
                true_state = cyborg.get_agent_state('True')
                remote.send(true_state)
                
            elif cmd == 'close':
                remote.close()
                break
                
        except EOFError:
            break


class ParallelEnvWrapper:
    """Vectorized environment wrapper for parallel execution"""
    
    def __init__(self, n_envs: int = 25, scenario_path: str = None, 
                 red_agent_type=RedMeanderAgent):
        """
        Initialize parallel environments
        
        Args:
            n_envs: Number of parallel environments
            scenario_path: Path to scenario file
            red_agent_type: Red agent class to use
        """
        self.n_envs = n_envs
        self.scenario_path = scenario_path or '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
        self.red_agent_type = red_agent_type
        
        # Create environment function
        def make_env():
            cyborg = CybORG(self.scenario_path, 'sim', 
                           agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            return env, cyborg
        
        # Create processes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.processes = []
        
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            process = Process(target=worker_process, 
                            args=(work_remote, remote, make_env))
            process.daemon = True
            process.start()
            self.processes.append(process)
        
        # Close work remotes
        for remote in self.work_remotes:
            remote.close()
        
        # Get observation space from first env
        self.remotes[0].send(('reset', None))
        obs = self.remotes[0].recv()
        self.observation_shape = obs.shape
        self.observation_space = type('obj', (object,), {'shape': obs.shape})()
        
    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, 
                                                 np.ndarray, List[Dict]]:
        """
        Step all environments in parallel
        
        Args:
            actions: List of actions for each environment
            
        Returns:
            observations: (n_envs, obs_dim) array
            rewards: (n_envs,) array
            dones: (n_envs,) array
            infos: List of info dicts
        """
        # Send actions to all environments
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        # Collect results
        results = [remote.recv() for remote in self.remotes]
        
        observations, rewards, dones, infos = zip(*results)
        
        return (np.array(observations), 
                np.array(rewards), 
                np.array(dones), 
                list(infos))
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        for remote in self.remotes:
            remote.send(('reset', None))
        
        observations = [remote.recv() for remote in self.remotes]
        return np.array(observations)
    
    def get_true_states(self) -> List[Dict]:
        """Get true states from all environments"""
        for remote in self.remotes:
            remote.send(('get_true_state', None))
        
        true_states = [remote.recv() for remote in self.remotes]
        return true_states
    
    def close(self):
        """Close all environments"""
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()


class ParallelTrajectoryBuffer:
    """Buffer for storing trajectories from parallel environments"""
    
    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self.reset()
    
    def reset(self):
        """Clear all buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, states, actions, rewards, dones, log_probs=None, values=None):
        """Add transitions from all environments"""
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        
        if log_probs is not None:
            self.log_probs.append(log_probs)
        if values is not None:
            self.values.append(values)
    
    def get_trajectories(self):
        """
        Get all trajectories as flat arrays for PPO update
        
        Returns:
            Dictionary with flattened trajectory data
        """
        # Stack along time dimension first
        states = np.array(self.states)  # (T, n_envs, obs_dim)
        actions = np.array(self.actions)  # (T, n_envs)
        rewards = np.array(self.rewards)  # (T, n_envs)
        dones = np.array(self.dones)  # (T, n_envs)
        
        # Reshape to (n_envs * T, ...)
        n_steps = states.shape[0]
        
        trajectories = {
            'states': states.reshape(-1, states.shape[-1]),
            'actions': actions.reshape(-1),
            'rewards': rewards.reshape(-1),
            'dones': dones.reshape(-1),
            'n_envs': self.n_envs,
            'n_steps': n_steps
        }
        
        if self.log_probs:
            log_probs = np.array(self.log_probs)
            trajectories['log_probs'] = log_probs.reshape(-1)
            
        if self.values:
            values = np.array(self.values)
            trajectories['values'] = values.reshape(-1)
        
        return trajectories
    
    def compute_returns(self, gamma: float = 0.99):
        """
        Compute discounted returns for each environment
        
        Args:
            gamma: Discount factor
            
        Returns:
            Array of returns (n_envs * T,)
        """
        rewards = np.array(self.rewards)  # (T, n_envs)
        dones = np.array(self.dones)  # (T, n_envs)
        
        n_steps, n_envs = rewards.shape
        
        # Compute returns for each environment separately
        returns = np.zeros_like(rewards)
        
        for env_idx in range(n_envs):
            running_return = 0
            for t in reversed(range(n_steps)):
                if dones[t, env_idx]:
                    running_return = 0
                running_return = rewards[t, env_idx] + gamma * running_return
                returns[t, env_idx] = running_return
        
        # Flatten
        return returns.reshape(-1)

