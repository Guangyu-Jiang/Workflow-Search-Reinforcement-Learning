# Workflow Search for Reinforcement Learning

Implementation of workflow-guided reinforcement learning using GP-UCB to search over workflow orderings. Contains experiments on **CAGE2 Cybersecurity Environment** and **Gridworld Planning**.

## Installation

```bash
# Core dependencies
pip install torch numpy scipy scikit-learn matplotlib seaborn

# For Gridworld experiments
pip install gym pygame stable-baselines3

# For CAGE2 experiments
# Follow CAGE2 installation instructions from the CAGE2 repository
```

## Running Experiments

### CAGE2 Cybersecurity Environment

```bash
cd CAGE-Challenge-2-Experiment

# Workflow search with GP-UCB (default: 20,000 episodes)
python workflow_rl/workflow_search_ppo.py --red-agent B_lineAgent

# Baselines (default: 20,000 episodes)
python baselines/train_ppo.py --n-workers 200 --red-agent B_lineAgent
python baselines/train_sac.py --n-workers 50 --red-agent B_lineAgent
python baselines/train_hierarchical_rl.py --n-workers 200 --red-agent B_lineAgent
```

**Command-line arguments**:
- `--total-episodes`: Number of episodes (default: 20000)
- `--red-agent`: Red agent type (`B_lineAgent`, `RedMeanderAgent`, `SleepAgent`)
- `--n-workers`: Number of parallel workers (default varies by script)
- `--gp-length-scale`: GP-UCB kernel length scale (default: 0.5)

### Gridworld Planning Environment

```bash
cd Grid-World-Planning-Experiment

# Workflow search (default: ~20,000 episodes)
python workflow_search_ppo.py --iterations 20 --updates-per-workflow 50

# Baselines
python train_ppo_maze.py          # PPO (default: 800 updates × 25 envs = 20,000 episodes)
python train_sac_maze.py          # SAC (default: 20,000 episodes)
python train_hrl_maze.py          # HRL (default: 800 updates × 25 envs = 20,000 episodes)
```

**Command-line arguments**:
- `--updates`: Number of PPO updates (for PPO/HRL, episodes = updates × num-envs)
- `--total-episodes`: Number of episodes (for SAC)
- `--num-envs`: Number of parallel environments (default: 25)

## Repository Structure

```
Workflow-Search-Reinforcement-Learning/
├── CAGE-Challenge-2-Experiment/
│   ├── workflow_rl/workflow_search_ppo.py  # Main workflow search
│   ├── baselines/                           # Baseline algorithms
│   └── README.md                            # CAGE2 documentation
│
└── Grid-World-Planning-Experiment/
    ├── workflow_search_ppo.py              # Main workflow search
    ├── train_ppo_maze.py                   # PPO baseline
    ├── train_sac_maze.py                   # SAC baseline
    ├── train_hrl_maze.py                   # HRL baseline
    └── core/                               # Environment implementations
```

## Default Hyperparameters

- **Default Episodes**: 20,000 for all experiments
- **Learning Rate**: 0.002 (CAGE2), 3e-4 (Gridworld)
- **Network Architecture**: 64→64 MLP (CAGE2), 128→128 MLP (Gridworld)

See individual script `--help` for full argument list.

---

**Repository**: https://github.com/Guangyu-Jiang/Workflow-Search-Reinforcement-Learning
