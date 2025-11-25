"""
Vectorized Environment Implementation
Runs multiple environments in a single process using NumPy for efficiency
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


class VectorizedCAGE2Envs:
    """
    Vectorized environment wrapper that runs multiple environments in single process
    Avoids IPC overhead by keeping everything in one process
    """
    
    def __init__(self, n_envs: int = 200, scenario_path: str = None,
                 red_agent_type=RedMeanderAgent):
        """
        Initialize vectorized environments
        
        Args:
            n_envs: Number of parallel environments
            scenario_path: Path to scenario file
            red_agent_type: Red agent class to use
        """
        self.n_envs = n_envs
        self.scenario_path = scenario_path or '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
        self.red_agent_type = red_agent_type
        
        # Create all environments in the same process
        print(f"Creating {n_envs} environments in single process...")
        self.envs = []
        self.cyborgs = []
        
        for i in range(n_envs):
            cyborg = CybORG(self.scenario_path, 'sim', 
                           agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            self.envs.append(env)
            self.cyborgs.append(cyborg)
            
            if (i + 1) % 50 == 0:
                print(f"  Created {i + 1}/{n_envs} environments")
        
        # Get observation shape
        self.observation_shape = self.envs[0].observation_space.shape
        self.observation_space = self.envs[0].observation_space
        obs_dim = self.observation_shape[0]
        
        # Pre-allocate arrays for efficiency
        self.observations = np.zeros((n_envs, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(n_envs, dtype=np.float32)
        self.dones = np.zeros(n_envs, dtype=bool)
        self.infos = [{} for _ in range(n_envs)]
        
        # Track which environments need reset
        self.needs_reset = np.zeros(n_envs, dtype=bool)
        
        print(f"Vectorized environments ready!")
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments (no IPC, all in same process!)
        
        Args:
            actions: Array of actions for each environment
            
        Returns:
            observations: (n_envs, obs_dim) array
            rewards: (n_envs,) array
            dones: (n_envs,) array
            infos: List of info dicts
        """
        # Step each environment (no IPC overhead!)
        for i in range(self.n_envs):
            # Auto-reset if needed
            if self.needs_reset[i]:
                obs = self.envs[i].reset()
                self.observations[i] = obs
                self.needs_reset[i] = False
            
            # Step environment
            obs, reward, done, info = self.envs[i].step(actions[i])
            
            # Store results directly (no serialization!)
            self.observations[i] = obs
            self.rewards[i] = reward
            self.dones[i] = done
            self.infos[i] = info
            
            # Mark for reset if done
            if done:
                self.needs_reset[i] = True
                # Reset immediately for next step
                obs = self.envs[i].reset()
                self.observations[i] = obs
        
        return (self.observations.copy(),
                self.rewards.copy(),
                self.dones.copy(),
                self.infos.copy())
    
    def reset(self, env_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Reset specified environments or all
        
        Args:
            env_ids: List of environment indices to reset (None = all)
            
        Returns:
            observations: Array of reset observations
        """
        if env_ids is None:
            env_ids = range(self.n_envs)
        
        for i in env_ids:
            obs = self.envs[i].reset()
            self.observations[i] = obs
            self.needs_reset[i] = False
        
        return self.observations.copy()
    
    def get_true_states(self) -> List[Dict]:
        """Get true states from all environments"""
        states = []
        for i in range(self.n_envs):
            state = self.cyborgs[i].get_agent_state('True')
            states.append(state)
        return states
    
    def close(self):
        """Clean up environments"""
        # Nothing special needed - all in same process
        pass
    
    def __del__(self):
        """Ensure cleanup"""
        self.close()


class BatchVectorizedCAGE2:
    """
    Optimized vectorized environments using batch processing
    Process environments in batches to optimize cache usage
    """
    
    def __init__(self, n_envs: int = 200, batch_size: int = 10,
                 scenario_path: str = None, red_agent_type=RedMeanderAgent):
        """
        Initialize batch-vectorized environments
        
        Args:
            n_envs: Total number of environments
            batch_size: Number of environments to process together
            scenario_path: Path to scenario file
            red_agent_type: Red agent class
        """
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.n_batches = n_envs // batch_size
        
        # Use the base vectorized implementation
        self.vec_env = VectorizedCAGE2Envs(n_envs, scenario_path, red_agent_type)
        
        # Inherit properties
        self.observation_shape = self.vec_env.observation_shape
        self.observation_space = self.vec_env.observation_space
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step environments in batches for better cache locality
        
        Args:
            actions: Array of actions
            
        Returns:
            Standard RL step returns
        """
        obs_dim = self.observation_shape[0]
        observations = np.zeros((self.n_envs, obs_dim), dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = []
        
        # Process in batches for better cache usage
        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            batch_slice = slice(start_idx, end_idx)
            
            # Process batch
            for i in range(start_idx, end_idx):
                if self.vec_env.needs_reset[i]:
                    obs = self.vec_env.envs[i].reset()
                    self.vec_env.observations[i] = obs
                    self.vec_env.needs_reset[i] = False
                
                obs, reward, done, info = self.vec_env.envs[i].step(actions[i])
                
                observations[i] = obs
                rewards[i] = reward
                dones[i] = done
                infos.append(info)
                
                if done:
                    self.vec_env.needs_reset[i] = True
                    obs = self.vec_env.envs[i].reset()
                    observations[i] = obs
        
        return observations, rewards, dones, infos
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        return self.vec_env.reset()
    
    def get_true_states(self) -> List[Dict]:
        """Get true states"""
        return self.vec_env.get_true_states()
    
    def close(self):
        """Clean up"""
        self.vec_env.close()


def test_vectorized_speed():
    """Test the speed of vectorized implementation"""
    print("="*60)
    print("VECTORIZED IMPLEMENTATION TEST")
    print("="*60)
    
    n_envs = 50  # Use fewer for quick test
    n_steps = 100
    
    print(f"\nConfiguration:")
    print(f"  Environments: {n_envs}")
    print(f"  Steps: {n_steps}")
    print(f"  Total transitions: {n_envs * n_steps}")
    
    # Test standard vectorized
    print("\n1. Standard Vectorized Implementation")
    print("-"*40)
    
    start = time.time()
    envs = VectorizedCAGE2Envs(n_envs=n_envs, red_agent_type=B_lineAgent)
    creation_time = time.time() - start
    print(f"  Creation time: {creation_time:.2f}s")
    
    # Run steps
    print("  Running steps...")
    start = time.time()
    
    observations = envs.reset()
    for step in range(n_steps):
        actions = np.random.randint(0, 145, n_envs)
        observations, rewards, dones, infos = envs.step(actions)
        
        if step % 20 == 0:
            print(f"    Step {step}/{n_steps}")
    
    step_time = time.time() - start
    throughput = (n_envs * n_steps) / step_time
    
    print(f"\n  Results:")
    print(f"    Total time: {step_time:.2f}s")
    print(f"    Time per step: {step_time/n_steps*1000:.1f}ms")
    print(f"    Throughput: {throughput:.0f} transitions/second")
    
    envs.close()
    
    # Test batch vectorized
    print("\n2. Batch Vectorized Implementation")
    print("-"*40)
    
    start = time.time()
    envs = BatchVectorizedCAGE2(n_envs=n_envs, batch_size=10, red_agent_type=B_lineAgent)
    creation_time = time.time() - start
    print(f"  Creation time: {creation_time:.2f}s")
    
    # Run steps
    print("  Running steps...")
    start = time.time()
    
    observations = envs.reset()
    for step in range(n_steps):
        actions = np.random.randint(0, 145, n_envs)
        observations, rewards, dones, infos = envs.step(actions)
        
        if step % 20 == 0:
            print(f"    Step {step}/{n_steps}")
    
    step_time = time.time() - start
    throughput_batch = (n_envs * n_steps) / step_time
    
    print(f"\n  Results:")
    print(f"    Total time: {step_time:.2f}s")
    print(f"    Time per step: {step_time/n_steps*1000:.1f}ms")
    print(f"    Throughput: {throughput_batch:.0f} transitions/second")
    
    envs.close()
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"  Original (Pipe IPC): ~180 transitions/second")
    print(f"  Vectorized: {throughput:.0f} transitions/second ({throughput/180:.1f}x)")
    print(f"  Batch Vectorized: {throughput_batch:.0f} transitions/second ({throughput_batch/180:.1f}x)")
    
    return throughput, throughput_batch


if __name__ == "__main__":
    test_vectorized_speed()
