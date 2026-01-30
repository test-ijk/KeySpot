# MultiWalker-HARL

A research framework for **Multi-Agent Reinforcement Learning (MARL)** focused on disturbance resilience, skill-based control, and LLM integration across multiple domains.

## Overview

This project implements a unified MARL training framework with support for:

- **Multi-Agent Hierarchical Actor-Critic Algorithms**: MAPPO, MATD3, MADDPG, and more
- **Disturbance Testing Framework**: Comprehensive tools for testing agent resilience under various disturbances
- **Skill-Based Learning**: Custom skill implementations and evaluation mechanisms
- **LLM Integration**: Natural language processing support for intelligent agent decision-making
- **Multi-Domain Environments**: Robotics (MultiWalker), Traffic Control (SUMO), Power Systems (MAPDN)

## Project Structure

```
2507-multiwalker-harl/
├── packages/
│   ├── HARL/          # Core MARL framework and algorithms
│   ├── SUMO/          # Traffic simulation environment
│   └── MAPDN/         # Power distribution network control
├── sources/
│   └── skill/         # Training, evaluation, and configuration
├── results/           # Experiment outputs and models
└── *.py              # Batch testing and evaluation scripts
```

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

```bash
# Install Python 3.8 (or 3.11+)
uv python install 3.8

# Install dependencies
uv sync

# Run training
uv run python -m sources.skill.code.train
```

### Manual Installation

Alternatively, install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model with default configuration:

```bash
# Using uv
uv run python -m sources.skill.code.train

# Using python directly
python -m sources.skill.code.train --config-name=pettingzoo_mw
```

### Evaluation

Evaluate a trained model:

```bash
uv run python -m sources.skill.code.eval --config-name=default
```

### Batch Disturbance Testing

Run comprehensive disturbance tests of multiwalker:

```bash
python batch_disturbance_test.py
```

This will test multiple agents with various disturbance magnitudes and generate detailed reports in `./results/batch_disturbance/`.

### Single Disturbance Test

Test a single disturbance scenario of multiwalker:

```bash
python test_single_disturbance.py
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for flexible configuration management.

### Configuration Structure

- **Global configs**: `sources/skill/1.config/global/`
- **Task configs**: `sources/skill/1.config/task/`
  - `train/`: Training configurations
  - `eval/`: Evaluation configurations
- **Algorithm configs**: `packages/HARL/harl/configs/algos_cfgs/`
- **Environment configs**: `packages/HARL/harl/configs/envs_cfgs/`

### Example Configuration

```yaml
# sources/skill/1.config/task/train/pettingzoo_mw.yaml
defaults:
  - algorithm: mappo
  - environment: pettingzoo_mw
  - scenario: move

wandb:
  wandb_project: mw_skill
  wandb_group: latest

model:
  save_group: latest
```

## Supported Environments

### 1. PettingZoo MultiWalker (`pettingzoo_mw`)
Multi-agent walker coordination with customizable disturbances and skills.

**Features:**
- Custom disturbance injection (adaptive, random modes)
- Skill-based movement modifications
- LLM integration for agent decisions
- Target-based coordination

### 2. SUMO Traffic Control (`sumo`, `sumo_llm`)
Traffic signal control and traffic flow management.

**Features:**
- Various network configurations (grids, intersections)
- Traffic event simulation (lane closures, incidents)
- LLM-based traffic management

### 3. MAPDN Power Control (`mapdn`)
Active voltage control in power distribution networks.

**Features:**
- Support for 33-bus, 141-bus, and 322-bus networks
- Distributed and decentralized control modes
- Traditional control methods for comparison

## Supported Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| MAPPO | On-Policy | Multi-Agent PPO |
| MATD3 | Off-Policy | Multi-Agent TD3 |
| MADDPG | Off-Policy | Multi-Agent DDPG |
| HAPPO | On-Policy | Hierarchical Actor-Critic PPO |
| HATD3 | Off-Policy | Hierarchical Actor-Critic TD3 |
| HADDPG | Off-Policy | Hierarchical Actor-Critic DDPG |
| HASAC | Off-Policy | Hierarchical Actor-Critic SAC |
| MAA2C | On-Policy | Multi-Agent Advantage Actor-Critic |
| HA2C | On-Policy | Hierarchical Actor-Critic A2C |

## Results and Logging

Training results are saved to:
- Models: `./results/models/`
- Outputs: `./results/outputs/`
- Batch tests: `./results/batch_disturbance/`

The project integrates with:
- **W&B**: Experiment tracking and visualization
- **TensorBoard**: Training metrics
- **Neptune**: Alternative experiment tracking

## Key Features

### Disturbance Testing Framework

Comprehensive disturbance testing with:
- Configurable disturbance types (magnitude-based, adaptive)
- Target agent selection
- Recovery time analysis
- Episode filtering and statistics
- Visualization tools

### LLM Integration

- Natural language explanations for agent decisions
- Target-based coordination through LLM prompts
- Configurable prompt templates

### Skill-Based Learning

- Custom skill implementations for movement
- Skill evaluation and comparison
- Disturbance-specific skill adaptation

## Development

### Code Structure

**Core Modules:**
- `harl/`: Main framework code
  - `algorithms/`: MARL algorithm implementations
  - `envs/`: Environment wrappers
  - `common/`: Shared utilities (buffers, loggers)

**Training Scripts:**
- `sources/skill/code/train.py`: Main training entry point
- `sources/skill/code/eval.py`: Evaluation script
- `sources/skill/code/run.py`: Multi-command execution

**Configuration:**
- `sources/skill/1.config/`: Hydra configurations
- `sources/skill/code/types/`: Type definitions for configs


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon:
- [HARL](https://github.com/PKU-MARL/HARL) - Hierarchical Actor-Critic Reinforcement Learning
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) - Multi-agent RL environments
- [SUMO](https://www.eclipse.org/sumo/) - Simulation of Urban MObility

## Contact

For questions and issues, please open a GitHub issue or contact [your-email@example.com].
