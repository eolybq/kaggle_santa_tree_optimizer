# Santa 2025 - The packing problem

This repository contains my solution for the [Santa 2025 - The packing problem](https://www.kaggle.com/competitions/santa-2025/overview) Kaggle competition. The goal is to pack between 1 and 200 distinct Christmas trees into the smallest possible square box without overlapping.

## Overview

The solution utilizes a two-stage approach: **Initialization** followed by **Optimization**.

### 1. Initialization (`services/initial_cords.py`)
Generates a valid starting configuration for each puzzle size ($N=1 \dots 200$).
- Uses a **greedy constructive heuristic**.
- Sequentially places trees at random coordinates $(x, y)$ and rotations.
- Checks for collisions using spatial indexing (`STRtree`).
- Accepts the first valid position found for each tree to ensure a feasible initial state.

### 2. Optimization (`optimize.py`)
Refines the valid initial solution to minimize the bounding box area.
- Implements a **Hill Climbing / Local Search** algorithm (structured similarly to Simulated Annealing).
- **Perturbation:** In each iteration, a random tree is selected, and its position and rotation are slightly perturbed using Gaussian noise.
- **Validation:** The new configuration is checked for collisions. If invalid, the move is rejected immediately.
- **Selection:** If valid, a loss function (based on distance between trees and distance to the origin) is calculated. The move is accepted if it improves the packing density.
- Uses high-precision arithmetic (`decimal` module) to satisfy competition submission requirements.

## Project Structure

- `optimize.py`: Main script for running the optimization loop.
- `services/chtree.py`: Defines the `ChristmasTree` geometry and polygon logic using `shapely`.
- `services/initial_cords.py`: Logic for generating initial valid placements.
- `data/`: Contains input and output CSV files.

## Dependencies

- Python 3.x
- `pandas`
- `numpy`
- `shapely`