# CAGE2 Workflow Search RL

A reinforcement learning system for finding optimal defense workflows in the CAGE2 cybersecurity environment using hierarchical RL with Gaussian Process Upper Confidence Bound (GP-UCB) search.

## Overview

This project implements a two-level reinforcement learning approach:

1. **High-level**: GP-UCB algorithm searches for optimal workflow orderings (permutations of unit types)
2. **Low-level**: PPO agent learns to execute a given workflow with alignment rewards

## Key Features

- **Workflow Conditioning**: PPO agent is conditioned on workflow priority orders
- **Normalized Alignment Rewards**: Rewards based on compliance rate, not fix count
- **Per-step Distribution**: Smooth alignment rewards distributed across episodes
- **Parallel Environments**: 25 parallel CAGE2 environments for stable training
- **Full Action Space**: Uses all 145 actions (no reduction)

## Architecture

```
Workflow Space (120 permutations)
    ↓ GP-UCB Search
Selected Workflow Order
    ↓ PPO Training
Aligned Defense Policy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Guangyu-Jiang/cage2-workflow-search-rl.git
cd cage2-workflow-search-rl
```

2. Set up the CAGE2 environment (see CAGE2 documentation)

3. Install dependencies:
```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn
```

## Usage

### Quick Start

Run the complete workflow search:

```bash
python workflow_rl/parallel_train_workflow_rl.py
```

### Configuration

Key parameters in `parallel_train_workflow_rl.py`:

- `n_envs`: Number of parallel environments (default: 25)
- `n_workflows`: Number of workflows to explore (default: 20)
- `train_episodes_per_env`: Episodes per environment (default: 2500)
- `alignment_alpha`: Compliance bonus weight (default: 10.0)
- `alignment_beta`: Violation penalty weight (default: 10.0)

### Testing

Run individual components:

```bash
# Test parallel environments
python workflow_rl/test_parallel_training.py

# Test compliance calculation
python workflow_rl/test_compliance_logic.py

# Test GP-UCB search
python workflow_rl/test_gpucb_output.py
```

## Workflow Representation

Workflows are represented as permutations of 5 unit types:
- `defender`: Most vulnerable, should be fixed first
- `enterprise`: Business-critical systems
- `op_server`: Operational servers
- `op_host`: Operational hosts
- `user`: User workstations

Example: `['defender', 'enterprise', 'op_server', 'op_host', 'user']`

## Reward Design

### Environment Reward
Original CAGE2 reward for minimizing damage and restoring systems.

### Alignment Reward
Normalized reward based on compliance rate:
- **Formula**: `α × compliance_rate - β × (1 - compliance_rate)`
- **Distribution**: Spread across all episode steps for smooth learning
- **Default**: α=10.0, β=10.0

## Results

The system learns to:
1. Follow prescribed workflow orders with high compliance (70-95%)
2. Balance environment performance with workflow adherence
3. Discover effective defense strategies through GP-UCB search

## Files Structure

```
workflow_rl/
├── parallel_train_workflow_rl.py    # Main training script
├── parallel_order_conditioned_ppo.py # PPO agent implementation
├── gp_ucb_order_search.py          # GP-UCB workflow search
├── order_based_workflow.py         # Workflow representation
├── parallel_env_wrapper.py         # Parallel environment wrapper
└── test_*.py                       # Test scripts
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the CAGE2 cybersecurity research initiative.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cage2_workflow_rl,
  title={CAGE2 Workflow Search RL},
  author={Guangyu Jiang},
  year={2024},
  url={https://github.com/Guangyu-Jiang/cage2-workflow-search-rl}
}
```