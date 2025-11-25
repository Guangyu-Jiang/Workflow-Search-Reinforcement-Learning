"""
Shared Memory Version of Parallel Environment Wrapper
Optimizes IPC by using shared memory instead of pipes for large data
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from multiprocessing import Process, Queue, shared_memory
import multiprocessing as mp
import queue
import time
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


class SharedMemoryEnvWorker:
    """Worker that uses shared memory for observations"""
    
    def __init__(self, worker_id: int, obs_shm_name: str, reward_shm_name: str, done_shm_name: str,
                 cmd_queue: Queue, result_queue: Queue, env_fn):
        """
        Initialize worker with shared memory
        
        Args:
            worker_id: Unique identifier for this worker
            obs_shm_name: Name of shared memory for observations
            reward_shm_name: Name of shared memory for rewards
            done_shm_name: Name of shared memory for done flags
            cmd_queue: Queue for receiving commands
            result_queue: Queue for sending small results
            env_fn: Function to create environment
        """
        self.worker_id = worker_id
        self.env, self.cyborg = env_fn()
        
        # Connect to shared memory
        self.obs_shm = shared_memory.SharedMemory(name=obs_shm_name)
        self.reward_shm = shared_memory.SharedMemory(name=reward_shm_name)
        self.done_shm = shared_memory.SharedMemory(name=done_shm_name)
        
        # Create numpy arrays backed by shared memory
        obs_shape = self.env.observation_space.shape[0]
        self.obs_array = np.ndarray((obs_shape,), dtype=np.float32, 
                                   buffer=self.obs_shm.buf[worker_id * obs_shape * 4:(worker_id + 1) * obs_shape * 4])
        
        self.reward_array = np.ndarray((1,), dtype=np.float32,
                                      buffer=self.reward_shm.buf[worker_id * 4:(worker_id + 1) * 4])
        
        self.done_array = np.ndarray((1,), dtype=np.bool_,
                                    buffer=self.done_shm.buf[worker_id:worker_id + 1])
        
        self.cmd_queue = cmd_queue
        self.result_queue = result_queue
    
    def run(self):
        """Main worker loop"""
        while True:
            try:
                cmd, data = self.cmd_queue.get(timeout=1)
                
                if cmd == 'step':
                    # Execute step
                    obs, reward, done, info = self.env.step(data)
                    
                    # Write directly to shared memory (no serialization!)
                    self.obs_array[:] = obs
                    self.reward_array[0] = reward
                    self.done_array[0] = done
                    
                    if done:
                        obs = self.env.reset()
                        self.obs_array[:] = obs
                    
                    # Only send small info dict through queue
                    self.result_queue.put((self.worker_id, info))
                    
                elif cmd == 'reset':
                    obs = self.env.reset()
                    self.obs_array[:] = obs
                    self.result_queue.put((self.worker_id, None))
                    
                elif cmd == 'get_true_state':
                    # True state still needs to go through queue (too complex for shared memory)
                    true_state = self.cyborg.get_agent_state('True')
                    self.result_queue.put((self.worker_id, true_state))
                    
                elif cmd == 'close':
                    # Clean up shared memory references
                    self.obs_shm.close()
                    self.reward_shm.close()
                    self.done_shm.close()
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                break


def worker_process_shm(worker_id, obs_shm_name, reward_shm_name, done_shm_name,
                       cmd_queue, result_queue, env_fn):
    """Process function for shared memory worker"""
    worker = SharedMemoryEnvWorker(worker_id, obs_shm_name, reward_shm_name, 
                                  done_shm_name, cmd_queue, result_queue, env_fn)
    worker.run()


class ParallelEnvSharedMemory:
    """Parallel environment wrapper using shared memory for efficiency"""
    
    def __init__(self, n_envs: int = 200, scenario_path: str = None,
                 red_agent_type=RedMeanderAgent):
        """
        Initialize parallel environments with shared memory
        
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
        
        # Get observation shape from a temp environment
        temp_env, _ = make_env()
        self.observation_shape = temp_env.observation_space.shape
        self.observation_space = temp_env.observation_space
        obs_dim = self.observation_shape[0]
        del temp_env
        
        # Create shared memory blocks
        self.obs_shm = shared_memory.SharedMemory(
            create=True, size=n_envs * obs_dim * 4)  # float32 = 4 bytes
        self.reward_shm = shared_memory.SharedMemory(
            create=True, size=n_envs * 4)  # float32 = 4 bytes
        self.done_shm = shared_memory.SharedMemory(
            create=True, size=n_envs)  # bool = 1 byte
        
        # Create numpy arrays backed by shared memory
        self.obs_array = np.ndarray((n_envs, obs_dim), dtype=np.float32, buffer=self.obs_shm.buf)
        self.reward_array = np.ndarray((n_envs,), dtype=np.float32, buffer=self.reward_shm.buf)
        self.done_array = np.ndarray((n_envs,), dtype=np.bool_, buffer=self.done_shm.buf)
        
        # Create command and result queues
        self.cmd_queues = [Queue() for _ in range(n_envs)]
        self.result_queue = Queue()
        
        # Start worker processes with staggered initialization for large counts
        self.processes = []
        batch_size = 50  # Start processes in batches
        
        if n_envs >= 50:
            print(f"  Starting {n_envs} parallel environments...")
        
        for i in range(n_envs):
            p = Process(target=worker_process_shm,
                       args=(i, self.obs_shm.name, self.reward_shm.name, 
                            self.done_shm.name, self.cmd_queues[i],
                            self.result_queue, make_env))
            p.daemon = True
            
            try:
                p.start()
                self.processes.append(p)
                
                # Add small delay every batch_size processes to avoid overwhelming system
                if (i + 1) % batch_size == 0:
                    import time
                    if n_envs >= 50:
                        print(f"    Started {i+1}/{n_envs} environments...")
                    time.sleep(0.1)  # 100ms pause every batch
                    
            except Exception as e:
                print(f"Error starting process {i}: {e}")
                # Clean up already started processes
                for proc in self.processes:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError(f"Failed to start worker process {i}: {e}")
        
        if n_envs >= 50:
            print(f"    ✓ All {n_envs} environments started successfully")
        
        # Initial reset
        self.reset()
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments in parallel using shared memory
        
        Args:
            actions: List of actions for each environment
            
        Returns:
            observations: (n_envs, obs_dim) array - read directly from shared memory
            rewards: (n_envs,) array - read directly from shared memory
            dones: (n_envs,) array - read directly from shared memory
            infos: List of info dicts
        """
        # Send actions to all workers
        for i, action in enumerate(actions):
            self.cmd_queues[i].put(('step', action))
        
        # Collect info dicts (small data through queue)
        infos = [None] * self.n_envs
        for _ in range(self.n_envs):
            worker_id, info = self.result_queue.get()
            infos[worker_id] = info
        
        # Return arrays directly from shared memory (no copy needed!)
        return (self.obs_array.copy(),  # Copy for safety
                self.reward_array.copy(),
                self.done_array.copy(),
                infos)
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        for i in range(self.n_envs):
            self.cmd_queues[i].put(('reset', None))
        
        # Wait for all resets to complete
        for _ in range(self.n_envs):
            self.result_queue.get()
        
        return self.obs_array.copy()
    
    def get_true_states(self) -> List[Dict]:
        """Get true states from all environments"""
        for i in range(self.n_envs):
            self.cmd_queues[i].put(('get_true_state', None))
        
        states = [None] * self.n_envs
        for _ in range(self.n_envs):
            worker_id, state = self.result_queue.get()
            states[worker_id] = state
        
        return states
    
    def close(self):
        """Clean up shared memory and processes"""
        # Send close command to all workers
        for i in range(self.n_envs):
            self.cmd_queues[i].put(('close', None))
        
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        
        # Clean up shared memory
        self.obs_shm.close()
        self.obs_shm.unlink()
        self.reward_shm.close()
        self.reward_shm.unlink()
        self.done_shm.close()
        self.done_shm.unlink()
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.close()
        except:
            pass


def test_shared_memory_speed():
    """Test the speed of shared memory implementation"""
    print("="*60)
    print("SHARED MEMORY IMPLEMENTATION TEST")
    print("="*60)
    
    n_envs = 50  # Use fewer for quick test
    n_steps = 100
    
    print(f"\nConfiguration:")
    print(f"  Environments: {n_envs}")
    print(f"  Steps: {n_steps}")
    print(f"  Total transitions: {n_envs * n_steps}")
    
    # Create environments
    print("\nCreating shared memory environments...")
    start = time.time()
    envs = ParallelEnvSharedMemory(n_envs=n_envs, red_agent_type=B_lineAgent)
    creation_time = time.time() - start
    print(f"  Created in {creation_time:.2f}s")
    
    # Run steps
    print("\nRunning steps...")
    start = time.time()
    
    observations = envs.reset()
    for step in range(n_steps):
        actions = np.random.randint(0, 145, n_envs)
        observations, rewards, dones, infos = envs.step(actions)
        
        if step % 20 == 0:
            print(f"  Step {step}/{n_steps}")
    
    step_time = time.time() - start
    
    # Calculate throughput
    total_transitions = n_envs * n_steps
    throughput = total_transitions / step_time
    
    print(f"\nResults:")
    print(f"  Total time: {step_time:.2f}s")
    print(f"  Time per step: {step_time/n_steps*1000:.1f}ms")
    print(f"  Throughput: {throughput:.0f} transitions/second")
    
    # Compare with original
    print(f"\nComparison:")
    print(f"  Original (Pipe): ~180 transitions/second")
    print(f"  Shared Memory: {throughput:.0f} transitions/second")
    print(f"  Speedup: {throughput/180:.1f}x")
    
    # Cleanup
    envs.close()
    
    return throughput


if __name__ == "__main__":
    test_shared_memory_speed()
