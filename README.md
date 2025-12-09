# Evolutionary Knapsack Problem Solver

A Python implementation of a Genetic Algorithm (GA) to solve the 0/1 Knapsack Problem using the DEAP (Distributed Evolutionary Algorithms in Python) framework. The project includes an interactive GUI application and a Jupyter notebook for experimentation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI Application](#gui-application)
  - [Jupyter Notebook](#jupyter-notebook)
- [Example Datasets](#example-datasets)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The **0/1 Knapsack Problem** is a classic optimization problem where you need to select items with given weights and values to maximize total value while staying within a weight capacity constraint. This project uses evolutionary algorithms to find near-optimal solutions.

## Features

- **Interactive GUI Application** - User-friendly Tkinter interface for configuring and running genetic algorithms
- **Custom Dataset Generator** - Generate knapsack problems with:
  - Heavy items (high value, high weight)
  - Efficient items (good value-to-weight ratio)
  - Deceptive items (appear valuable but aren't optimal)
- **Pre-loaded Example Datasets** - Four realistic scenarios:
  - Groceries shopping
  - Household items
  - Magic items (fantasy theme)
  - Tech tools
- **Real-time Visualization** - Live fitness evolution plots during algorithm execution
- **Configurable GA Parameters**:
  - Population size
  - Number of generations
  - Crossover rate
  - Mutation rate
  - Tournament selection size
  - Elitism option
- **Detailed Results View** - Tabular display of selected items with weights and values
- **Jupyter Notebook** - For research and experimentation

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/haarmeggido/evolutionary-knapsack-problem.git
cd evolutionary-knapsack-problem
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### GUI Application

Launch the graphical interface:

```bash
python src/main.py
```

**Using the GUI:**

1. **Load a Dataset:**
   - Use `File → Load Example` to select a pre-loaded scenario (Groceries, Household, Magic Items, Tech Tools)
   - Or use `File → New Generated Problem` to create a random dataset

2. **Configure Parameters:**
   - Adjust dataset parameters (number of items, capacity ratio, etc.)
   - Configure genetic algorithm settings (population size, generations, mutation rate, etc.)

3. **Run the Algorithm:**
   - Click "Run Genetic Algorithm"
   - Watch the fitness evolution plot the graph of found solution

4. **View Results:**
   - Check the solution summary in the left panel
   - Open detailed results window to see which items were selected

### Jupyter Notebook

For experimentation and research:

```bash
jupyter notebook src/Evolutionary_Algorithms_Knapsack_problem.ipynb
```

The notebook provides step-by-step examples and allows you to customize the algorithm further.

Here is a clean replacement section for your **README.md**, rewritten as a summary table:


## Example Datasets

The project includes several curated JSON datasets located in `data/examples/`.
Each dataset contains item names, weights, and values, suitable for testing the Genetic Algorithm under different themes.

### Dataset Summary (evaluated at default **0.4 capacity ratio**)

| Dataset         | Optimal Value | Optimal Weight | # Items |
| --------------- | ------------- | -------------- | ------- |
| **Groceries**   | 283           | 9.70           | 30      |
| **Household**   | 749           | 31.30          | 30      |
| **Magic Items** | 45 700        | 19.20          | 30      |
| **Tech Tools**  | 5 011         | 18.31          | 14      |

## Configuration

### Dataset Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_items` | 60 | Total number of items to generate |
| `capacity_ratio` | 0.3 | Knapsack capacity as ratio of total weight |
| `n_heavy` | 10 | Number of heavy but valuable items |
| `n_efficient` | 8 | Number of items with good value/weight ratio |
| `n_deceptive` | 5 | Number of items that appear valuable but aren't |
| `seed` | None | Random seed for reproducibility |

### Genetic Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 50 | Number of individuals in population |
| `generations` | 150 | Number of generations to evolve |
| `crossover_rate` | 0.8 | Probability of crossover between parents |
| `mutation_rate` | 0.05 | Probability of bit flip mutation |
| `tournament_size` | 3 | Number of individuals in tournament selection |
| `elitism` | True | Keep best individual from previous generation |

## How It Works

### Genetic Algorithm Overview

1. **Initialization**: Random binary chromosomes (0/1 for each item)
2. **Fitness Evaluation**: Calculate total value with penalty for exceeding capacity
3. **Selection**: Tournament selection to choose parents
4. **Crossover**: One-point crossover to create offspring
5. **Mutation**: Bit-flip mutation to maintain diversity
6. **Elitism**: Preserve best solution across generations
7. **Iteration**: Repeat for specified number of generations

### Fitness Function

```python
fitness = total_value - penalty * (total_weight - capacity)  # if overweight
fitness = total_value  # if within capacity
```

The penalty ensures solutions evolve toward feasible (within capacity) solutions while maximizing value. Penalty is updated dynamically to fit the crrent problem, as the average ratio of weight to the cost

## Project Structure

```
evolutionary-knapsack-problem/
├── assets/                           # Application icons
│   ├── icon_boring.ico
│   ├── icon_fun.ico
│   └── icon_gene.ico
├── data/
│   └── examples/                     # Pre-loaded example datasets
│       ├── groceries.json
│       ├── household.json
│       ├── magic_items.json
│       └── tech_tools.json
├── src/
│   ├── core/
│   │   ├── dataset_generator.py      # Random dataset generation
│   │   ├── genetic_algorithm.py      # GA implementation (DEAP)
│   │   └── knapsack_problem.py       # Problem definition & fitness
│   ├── ui/
│   │   └── gui.py                    # Tkinter GUI application
│   ├── utils/
│   │   └── plot_utils.py             # Matplotlib plotting helpers
│   ├── main.py                       # Application entry point
│   └── Evolutionary_Algorithms_Knapsack_problem.ipynb
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Created with ❤️ using Python, NumPy, Matplotlib, DEAP, and Tkinter**
