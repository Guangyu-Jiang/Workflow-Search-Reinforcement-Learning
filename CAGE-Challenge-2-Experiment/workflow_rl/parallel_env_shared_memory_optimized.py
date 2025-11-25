"""
Optimized Parallel Environment with Efficient True State Handling
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Process, Queue, shared_memory, Pipe
import multiprocessing as mp
import queue
import time
import pickle
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


class OptimizedEnvWorker:
    """Worker with dedicated pipe for true states and cached states"""
    
    def __init__(self, worker_id: int, obs_shm_name: str, reward_shm_name: str, 
                 done_shm_name: str, cmd_queue: Queue, state_pipe, env_fn):
        """
        Initialize worker with dedicated state pipe
        
        Args:
            state_pipe: Dedicated pipe for true state communication (no queue contention!)
        """
        self.worker_id = worker_id
        self.env, self.cyborg = env_fn()
        
        # Connect to shared memory for fast data
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
        self.state_pipe = state_pipe  # Dedicated pipe for true states
        
        # Cache for true state (only update when needed)
        self.cached_true_state = None
        self.cache_step_count = 0
        self.cache_update_interval = 10  # Only update true state every 10 steps
    
    def run(self):
        """Main worker loop"""
        while True:
            try:
                cmd, data = self.cmd_queue.get(timeout=1)
                
                if cmd == 'step':
                    # Execute step
                    obs, reward, done, info = self.env.step(data)
                    
                    # Write directly to shared memory
                    self.obs_array[:] = obs
                    self.reward_array[0] = reward
                    self.done_array[0] = done
                    
                    if done:
                        obs = self.env.reset()
                        self.obs_array[:] = obs
                        # Update cache on episode end
                        self.cached_true_state = None
                        self.cache_step_count = 0
                    else:
                        self.cache_step_count += 1
                    
                    # Send simple ACK through pipe
                    self.state_pipe.send(('step_done', None))
                    
                elif cmd == 'reset':
                    obs = self.env.reset()
                    self.obs_array[:] = obs
                    self.cached_true_state = None
                    self.cache_step_count = 0
                    self.state_pipe.send(('reset_done', None))
                    
                elif cmd == 'get_true_state':
                    # Use cached state if available and recent
                    if self.cached_true_state is None or self.cache_step_count >= self.cache_update_interval:
                        self.cached_true_state = self.cyborg.get_agent_state('True')
                        self.cache_step_count = 0
                    
                    # Send through dedicated pipe (no contention!)
                    self.state_pipe.send(('true_state', self.cached_true_state))
                    
                elif cmd == 'get_true_state_if_needed':
                    # Only get state at episode boundaries or every N steps
                    if self.done_array[0] or self.cache_step_count >= self.cache_update_interval:
                        true_state = self.cyborg.get_agent_state('True')
                        self.cached_true_state = true_state
                    else:
                        true_state = None  # Signal that update isn't needed
                    
                    self.state_pipe.send(('true_state_sparse', true_state))
                    
                elif cmd == 'close':
                    self.obs_shm.close()
                    self.reward_shm.close()
                    self.done_shm.close()
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                break


def worker_process_optimized(worker_id, obs_shm_name, reward_shm_name, done_shm_name,
                             cmd_queue, state_pipe, env_fn):
    """Process function for optimized worker"""
    worker = OptimizedEnvWorker(worker_id, obs_shm_name, reward_shm_name, 
                                done_shm_name, cmd_queue, state_pipe, env_fn)
    worker.run()


class ParallelEnvSharedMemoryOptimized:
    """Optimized parallel environment with efficient true state handling"""
    
    def __init__(self, n_envs: int = 100, scenario_path: str = None,
                 red_agent_type=RedMeanderAgent, sparse_true_states: bool = True):
        """
        Initialize optimized parallel environments
        
        Args:
            sparse_true_states: If True, only get true states at episode boundaries
        """
        self.n_envs = n_envs
        self.scenario_path = scenario_path or '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
        self.red_agent_type = red_agent_type
        self.sparse_true_states = sparse_true_states
        
        # Create environment function
        def make_env():
            cyborg = CybORG(self.scenario_path, 'sim', 
                           agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            return env, cyborg
        
        # Get observation shape
        temp_env, _ = make_env()
        self.observation_shape = temp_env.observation_space.shape
        self.observation_space = temp_env.observation_space
        obs_dim = self.observation_shape[0]
        del temp_env
        
        # Create shared memory blocks
        self.obs_shm = shared_memory.SharedMemory(
            create=True, size=n_envs * obs_dim * 4)
        self.reward_shm = shared_memory.SharedMemory(
            create=True, size=n_envs * 4)
        self.done_shm = shared_memory.SharedMemory(
            create=True, size=n_envs)
        
        # Create numpy arrays backed by shared memory
        self.obs_array = np.ndarray((n_envs, obs_dim), dtype=np.float32, buffer=self.obs_shm.buf)
        self.reward_array = np.ndarray((n_envs,), dtype=np.float32, buffer=self.reward_shm.buf)
        self.done_array = np.ndarray((n_envs,), dtype=np.bool_, buffer=self.done_shm.buf)
        
        # Create command queues and DEDICATED pipes for each worker
        self.cmd_queues = [Queue() for _ in range(n_envs)]
        self.state_pipes = []
        self.worker_conns = []
        
        # Create dedicated pipes for true state communication (no contention!)
        for i in range(n_envs):
            parent_conn, worker_conn = Pipe()
            self.state_pipes.append(parent_conn)
            self.worker_conns.append(worker_conn)
        
        # Start worker processes
        self.processes = []
        
        if n_envs >= 50:
            print(f"  Starting {n_envs} optimized parallel environments...")
        
        for i in range(n_envs):
            p = Process(target=worker_process_optimized,
                       args=(i, self.obs_shm.name, self.reward_shm.name, 
                            self.done_shm.name, self.cmd_queues[i],
                            self.worker_conns[i], make_env))
            p.daemon = True
            p.start()
            self.processes.append(p)
            
            # Stagger for large counts
            if (i + 1) % 50 == 0 and n_envs >= 50:
                print(f"    Started {i+1}/{n_envs} environments...")
                time.sleep(0.05)  # Shorter delay
        
        if n_envs >= 50:
            print(f"    ✓ All {n_envs} optimized environments ready")
        
        # Cache for true states
        self.cached_true_states = [None] * n_envs
        self.steps_since_true_state = [0] * n_envs
        
        # Initial reset
        self.reset()
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments (true states handled separately)"""
        # Send step commands
        for i, action in enumerate(actions):
            self.cmd_queues[i].put(('step', action))
            self.steps_since_true_state[i] += 1
        
        # Wait for completion (simple ACKs through pipes)
        for i in range(self.n_envs):
            msg_type, _ = self.state_pipes[i].recv()
            assert msg_type == 'step_done'
        
        # Return arrays directly from shared memory
        return (self.obs_array.copy(),
                self.reward_array.copy(),
                self.done_array.copy(),
                [{}] * self.n_envs)  # Info dicts if needed
    
    def get_true_states(self) -> List[Dict]:
        """Get true states efficiently using dedicated pipes"""
        if self.sparse_true_states:
            # Only get states where episodes ended or every N steps
            states = []
            for i in range(self.n_envs):
                if self.done_array[i] or self.steps_since_true_state[i] >= 10:
                    self.cmd_queues[i].put(('get_true_state', None))
                    msg_type, state = self.state_pipes[i].recv()
                    self.cached_true_states[i] = state
                    self.steps_since_true_state[i] = 0
                states.append(self.cached_true_states[i])
        else:
            # Get all states in parallel (but through dedicated pipes)
            for i in range(self.n_envs):
                self.cmd_queues[i].put(('get_true_state', None))
            
            # Receive through dedicated pipes (no contention!)
            states = []
            for i in range(self.n_envs):
                msg_type, state = self.state_pipes[i].recv()
                states.append(state)
                self.cached_true_states[i] = state
        
        return states
    
    def get_true_states_batch(self, indices: List[int]) -> Dict[int, Dict]:
        """Get true states only for specific environments"""
        for i in indices:
            self.cmd_queues[i].put(('get_true_state', None))
        
        states = {}
        for i in indices:
            msg_type, state = self.state_pipes[i].recv()
            states[i] = state
        
        return states
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        for i in range(self.n_envs):
            self.cmd_queues[i].put(('reset', None))
        
        for i in range(self.n_envs):
            msg_type, _ = self.state_pipes[i].recv()
            assert msg_type == 'reset_done'
            self.steps_since_true_state[i] = 0
        
        return self.obs_array.copy()
    
    def close(self):
        """Clean up"""
        for i in range(self.n_envs):
            self.cmd_queues[i].put(('close', None))
        
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
