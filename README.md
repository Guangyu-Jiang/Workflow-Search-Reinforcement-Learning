# Workflow Search for Reinforcement Learning

A comprehensive implementation of workflow-guided reinforcement learning for sequential decision-making tasks. This repository contains experiments on two domains: **CAGE2 Cybersecurity Environment** and **Gridworld Planning**, demonstrating how Gaussian Process Upper Confidence Bound (GP-UCB) can efficiently search over workflow orderings while learning optimal policies.

## 🎯 Overview

This project explores a two-level reinforcement learning framework:

1. **High-level Search**: GP-UCB algorithm searches for optimal workflow orderings (permutations of task priorities)
2. **Low-level Learning**: RL agents (PPO, SAC, HRL) learn to execute policies conditioned on workflow specifications

The key insight is that many sequential tasks have inherent ordering constraints (e.g., fixing critical systems before operational ones, visiting checkpoints in a specific order), and discovering these optimal orderings can dramatically improve learning efficiency.

## 📁 Repository Structure

```
Workflow-Search-Reinforcement-Learning/
├── CAGE-Challenge-2-Experiment/     # Cybersecurity domain experiments
│   ├── workflow_rl/                 # Core workflow search implementation
│   │   ├── workflow_search_ppo.py  # Main workflow search script
│   │   ├── gp_ucb_order_search.py  # GP-UCB search algorithm
│   │   └── order_based_workflow.py # Workflow representation
│   ├── baselines/                   # Baseline algorithms
│   │   ├── train_ppo.py            # PPO baseline
│   │   ├── train_sac.py            # SAC baseline
│   │   └── train_hierarchical_rl.py # HRL baseline
│   ├── Agents/                      # Agent implementations
│   ├── Wrappers/                    # Environment wrappers
│   └── README.md                    # CAGE2-specific documentation
│
├── Grid-World-Planning-Experiment/  # Gridworld navigation experiments
│   ├── core/                        # Environment implementations
│   │   ├── obstacle_maze_env.py    # Main maze environment
│   │   ├── workflow_alignment_system.py
│   │   └── ...
│   ├── workflow_search_ppo.py      # GP-UCB workflow search
│   ├── train_ppo_maze.py           # PPO baseline
│   ├── train_sac_maze.py           # SAC baseline
│   ├── train_hrl_maze.py           # HRL baseline
│   └── requirements.txt             # Dependencies
│
├── README.md                        # This file
├── requirements.txt                 # Core dependencies
└── .gitignore                       # Git ignore patterns
```

## 🔬 Experiments

### 1. CAGE2 Cybersecurity Environment

**Domain**: Network defense in the CAGE2 cybersecurity simulator

**Task**: Learn optimal defense workflows for fixing compromised systems. The agent must decide which unit types (defender, enterprise, operational servers, operational hosts, user workstations) to prioritize when multiple systems are compromised.

**Workflow Space**: 120 permutations of 5 unit types

**Key Features**:
- Order-conditioned PPO with alignment rewards
- Parallel episode collection (50-200 workers)
- Compliance-based reward shaping
- Full 145-action space (no reduction)

**Main Scripts**:
- `workflow_rl/workflow_search_ppo.py` - GP-UCB workflow search with PPO
- `baselines/train_ppo.py` - Standard PPO baseline
- `baselines/train_sac.py` - SAC baseline
- `baselines/train_hierarchical_rl.py` - Hierarchical RL (Options Framework) baseline

**Hyperparameters**:
- **Network Architecture**: 64→64 Tanh MLP (Actor-Critic)
- **Learning Rate**: 0.002 (Adam optimizer)
- **PPO**: γ=0.99, K_epochs=6, ε_clip=0.2
- **GP-UCB**: β=2.0, length_scale=0.5 (RBF kernel)
- **Default Episodes**: 20,000 per experiment

See [CAGE-Challenge-2-Experiment/README.md](CAGE-Challenge-2-Experiment/README.md) for detailed documentation.

### 2. Gridworld Planning Environment

**Domain**: Navigation with obstacles on a 30×30 grid

**Task**: Visit four checkpoint regions in a specified order while navigating around randomly placed obstacles.

**Workflow Space**: Permutations of checkpoint visit order [CP0, CP1, CP2, CP3]

**Key Features**:
- Multiple environment variants (obstacle maze, diagonal regions, diagonal corners)
- Strict workflow-based reward structure
- Milestone-based workflow representation
- Comparison with standard RL baselines

**Main Scripts**:
- `workflow_search_ppo.py` - GP-UCB workflow search
- `train_ppo_maze.py` - PPO baseline
- `train_sac_maze.py` - SAC baseline
- `train_hrl_maze.py` - Hierarchical RL baseline

**Hyperparameters**:
- **Network Architecture**: 128→128 ReLU MLP
- **Learning Rate**: 3e-4 (Adam optimizer)
- **PPO**: γ=0.99, K_epochs=4, ε_clip=0.2
- **Default Episodes**: 20,000 per experiment (800 updates × 25 envs)

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch numpy scipy scikit-learn matplotlib seaborn

# For Gridworld experiments
pip install gym pygame stable-baselines3

# For CAGE2 experiments
# Follow CAGE2 installation instructions at CAGE2 repository
```

### Running Experiments

#### CAGE2 Experiment

```bash
cd CAGE-Challenge-2-Experiment

# Run workflow search with GP-UCB (default: 20,000 episodes)
python workflow_rl/workflow_search_ppo.py \
    --red-agent B_lineAgent \
    --gp-length-scale 0.5

# Run PPO baseline (default: 20,000 episodes)
python baselines/train_ppo.py \
    --n-workers 200 \
    --red-agent B_lineAgent

# Run SAC baseline (default: 20,000 episodes)
python baselines/train_sac.py \
    --n-workers 50 \
    --red-agent B_lineAgent

# Run HRL baseline (default: 20,000 episodes)
python baselines/train_hierarchical_rl.py \
    --n-workers 200 \
    --red-agent B_lineAgent
```

#### Gridworld Experiment

```bash
cd Grid-World-Planning-Experiment

# Run workflow search (default: ~20,000 episodes)
python workflow_search_ppo.py \
    --iterations 20 \
    --updates-per-workflow 50

# Run PPO baseline (default: 20,000 episodes = 800 updates × 25 envs)
python train_ppo_maze.py \
    --updates 800 \
    --num-envs 25

# Run SAC baseline (default: 20,000 episodes)
python train_sac_maze.py \
    --n-workers 50

# Run HRL baseline (default: 20,000 episodes = 800 updates × 25 envs)
python train_hrl_maze.py \
    --updates 800 \
    --num-envs 25
```

**Note**: All experiments default to 20,000 episodes. You can override this with command-line arguments:
- CAGE2: `--total-episodes <number>`
- Gridworld (PPO/HRL): `--updates <number>` (episodes = updates × num-envs)
- Gridworld (SAC): `--total-episodes <number>`

## 📊 Key Results

### CAGE2 Environment

- **Workflow Search (GP-UCB-WS)**: Discovers effective defense workflows, achieving higher compliance rates (70-95%) compared to random orderings
- **Baseline Comparison**: Outperforms standard PPO, SAC, and HRL baselines in cumulative reward over 20,000 episodes
- **Workflow Conditioning**: Agents trained with workflow conditioning learn faster and achieve better final performance

### Gridworld Environment

- **Convergence**: GP-UCB-WS converges to optimal checkpoint ordering faster than exhaustive search
- **Baseline Comparison**: Demonstrates improved sample efficiency compared to standard RL methods
- **Generalization**: Learned workflows generalize across different obstacle configurations

## 🔧 Key Components

### Workflow Representation

Workflows are represented as permutations of task priorities:

- **CAGE2**: `['defender', 'enterprise', 'op_server', 'op_host', 'user']`
- **Gridworld**: `[0, 1, 2, 3]` (checkpoint visit order)

### GP-UCB Search

Uses Gaussian Process Upper Confidence Bound to efficiently explore workflow space:
- **Kernel**: RBF (Radial Basis Function) with configurable length scale
- **Distance Metric**: Kendall tau distance for permutation space
- **Beta/Kappa**: Controls exploration-exploitation tradeoff (default: 2.0-4.0)

### RL Algorithms

- **PPO**: Proximal Policy Optimization with workflow conditioning
- **SAC**: Soft Actor-Critic (off-policy baseline)
- **HRL**: Hierarchical RL with Options Framework

## 📈 Visualization

Results can be visualized using Jupyter notebooks:
- Check experiment log directories for CSV files with training metrics
- Use matplotlib/seaborn to plot convergence curves
- Compare workflow search vs. baseline performance

## 🧪 Reproducibility

All experiments include:
- Random seed control for reproducibility (default seeds specified in scripts)
- Experiment configuration JSON files saved in log directories
- CSV training logs with detailed metrics
- Hyperparameter documentation in README files

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{workflow_search_rl,
  title={Workflow Search for Reinforcement Learning},
  author={Guangyu Jiang},
  year={2024},
  url={https://github.com/Guangyu-Jiang/Workflow-Search-Reinforcement-Learning}
}
```

## 📄 License

This project is part of research on workflow-guided reinforcement learning. See individual experiment directories for specific license information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Additional Resources

- **CAGE2 Documentation**: See [CAGE-Challenge-2-Experiment/README.md](CAGE-Challenge-2-Experiment/README.md)
- **Gridworld Environment**: See [Grid-World-Planning-Experiment/core/obstacle_maze_env.py](Grid-World-Planning-Experiment/core/obstacle_maze_env.py)
- **Workflow Search Algorithm**: See [CAGE-Challenge-2-Experiment/workflow_rl/gp_ucb_order_search.py](CAGE-Challenge-2-Experiment/workflow_rl/gp_ucb_order_search.py)

---

**Repository**: https://github.com/Guangyu-Jiang/Workflow-Search-Reinforcement-Learning
