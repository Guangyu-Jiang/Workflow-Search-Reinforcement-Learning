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
│   ├── baselines/                   # Baseline algorithms (PPO, SAC, HRL)
│   ├── train_*.py                   # Training scripts
│   ├── plot.ipynb                   # Results visualization
│   └── README.md                    # CAGE2-specific documentation
│
├── Gird-World-Planning-Experiment/  # Gridworld navigation experiments
│   ├── core/                        # Environment implementations
│   ├── notebooks/                   # Analysis notebooks
│   ├── train_*.py                   # Training scripts
│   └── requirements.txt             # Dependencies
│
└── README.md                        # This file
```

## 🔬 Experiments

### 1. CAGE2 Cybersecurity Environment

**Domain**: Network defense in the CAGE2 cybersecurity simulator

**Task**: Learn optimal defense workflows for fixing compromised systems. The agent must decide which unit types (defender, enterprise, operational servers, operational hosts, user workstations) to prioritize when multiple systems are compromised.

**Workflow Space**: 120 permutations of 5 unit types

**Key Features**:
- Order-conditioned PPO with alignment rewards
- Parallel episode collection (200 workers)
- Compliance-based reward shaping
- Full 145-action space (no reduction)

**Main Scripts**:
- `workflow_rl/workflow_search_ppo.py` - GP-UCB workflow search with PPO
- `train_parallel_baseline.py` - Standard PPO baseline
- `train_parallel_sac.py` - SAC baseline
- `train_hierarchical_general.py` - Hierarchical RL (Options Framework) baseline

**Hyperparameters**:
- **Network Architecture**: 64→64 Tanh MLP (Actor-Critic)
- **Learning Rate**: 0.002 (Adam optimizer)
- **PPO**: γ=0.99, K_epochs=6, ε_clip=0.2
- **GP-UCB**: β=2.0, length_scale=0.5 (RBF kernel)

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
- `workflow_search_gpucb_maze_fixed.py` - GP-UCB workflow search
- `train_ppo_baseline_maze_mp.py` - PPO baseline
- `train_sac_baseline_maze.py` - SAC baseline
- `train_hrl_baseline_maze.py` - Hierarchical RL baseline

See [Gird-World-Planning-Experiment/environment_description.md](Gird-World-Planning-Experiment/environment_description.md) for environment details.

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch numpy scipy scikit-learn matplotlib seaborn

# For Gridworld experiments
pip install gym pygame

# For CAGE2 experiments
# Follow CAGE2 installation instructions
```

### Running Experiments

#### CAGE2 Experiment

```bash
cd CAGE-Challenge-2-Experiment

# Run workflow search with GP-UCB
python workflow_rl/workflow_search_ppo.py \
    --total_episodes 100000 \
    --episodes_per_update 200 \
    --red_agent B_lineAgent \
    --gp_length_scale 0.5

# Run PPO baseline
python train_parallel_baseline.py \
    --n_workers 200 \
    --total_episodes 100000 \
    --red_agent B_lineAgent
```

#### Gridworld Experiment

```bash
cd Gird-World-Planning-Experiment

# Run workflow search
python workflow_search_gpucb_maze_fixed.py \
    --total_episodes 50000 \
    --gp_beta 2.0

# Run PPO baseline
python train_ppo_baseline_maze_mp.py \
    --total_episodes 50000
```

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
- **Beta**: Controls exploration-exploitation tradeoff (default: 2.0)

### RL Algorithms

- **PPO**: Proximal Policy Optimization with workflow conditioning
- **SAC**: Soft Actor-Critic (off-policy baseline)
- **HRL**: Hierarchical RL with Options Framework

## 📈 Visualization

Results are available in:
- `CAGE-Challenge-2-Experiment/plot.ipynb` - CAGE2 convergence plots and tables
- `Gird-World-Planning-Experiment/notebooks/plot_baseline_vs_gpucb.ipynb` - Gridworld comparison plots
- `figures/` - Generated PDF/PNG figures

## 🧪 Reproducibility

All experiments include:
- Random seed control for reproducibility
- Experiment configuration JSON files
- CSV training logs with detailed metrics
- Hyperparameter documentation in README files

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{workflow_search_rl,
  title={Workflow Search for Reinforcement Learning},
  author={Guangyu Jiang},
  year={2024},
  url={https://github.com/yourusername/Workflow-Search-Reinforcement-Learning}
}
```

## 📄 License

This project is part of research on workflow-guided reinforcement learning. See individual experiment directories for specific license information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Additional Resources

- **CAGE2 Documentation**: See [CAGE-Challenge-2-Experiment/README.md](CAGE-Challenge-2-Experiment/README.md)
- **Gridworld Details**: See [Gird-World-Planning-Experiment/environment_description.md](Gird-World-Planning-Experiment/environment_description.md)
- **Workflow Selection**: See [CAGE-Challenge-2-Experiment/WORKFLOW_SELECTION_EXPLAINED.md](CAGE-Challenge-2-Experiment/WORKFLOW_SELECTION_EXPLAINED.md)

---

**Note**: The Gridworld experiment directory is named `Gird-World-Planning-Experiment` (with a typo) for historical reasons. The code is fully functional regardless of the directory name.

