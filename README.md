# Cooperative Multi-Agent Maze Escape

## Overview

This project implements a cooperative multi-agent environment where two agents collaborate to escape a maze with barriers. The environment encourages the development of cooperative strategies, requiring agents to navigate through the grid, avoid walls, and coordinate to open a door to escape successfully.

## Repository Structure

- **saved-models/**: Directory with pre-trained models.
- **AI models project.ipynb**: Jupyter notebook with the final training process.
- **EscapeEnv.py**: Defines the environment, rewards, rendering and so on.
- **PPO_agents.py**: Implements the PPO (Proximal Policy Optimization) learning algorithm.
- **actor_critic.py**: The actor and critic networks used in PPO.
- **evaluate_plot_save.py**: Additional functions for evaluating, plotting, saving and loading models.

## Environment Description

The environment is a cooperative multi-agent maze where two agents must work together to escape.
- Agents start from two different corners of the maze.
- Agents take turns moving within the grid.
- Agents must avoid walls and coordinate to open a door.
- The objective is to escape the maze by stepping on the final cells together.
  
---
## Results
You can see the complete training results in the Jupyter notebook. 
ALso you can read the report below with a showt overview of the project 

## GiFs!!
