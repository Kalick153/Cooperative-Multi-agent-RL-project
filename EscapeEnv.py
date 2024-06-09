import numpy as np
import pygame  # for rendering
import random
import functools
from gymnasium import spaces
from copy import copy
from pettingzoo import AECEnv  # step by step multi-agent env
from pettingzoo.utils import agent_selector


class Escape(AECEnv):
    # it is my first time, when I comment my code in the documentation style. I hope everything will be clear!

    """
    This module defines a cooperative multi-agent environment.
    In this environment, two agents try to escape together from a maze with barriers.

    Both agents are identical, starting from two different corners of the maze.

    The environment operates in a step-by-step manner:
    - Each agent takes turns to move in the grid.
    - The agents need to navigate through the grid, avoiding walls and coordinating to open a door.

    This setup encourages the agents to develop cooperative strategies to successfully escape the grid.
    """
    metadata = {"render_modes": ["human"], "name": "escape_v0"}

    def __init__(self, grid_size, random_gate=True, switched_finish=False, additional_wall=False, render_mode=None,
                 reward_rates={'penalty_per_move_per_row': -0.05, 'penalty_illegal': -1, 'penalty_closed_door': -3, 'reward_open_door': 300,
                               'reward_escape': 500, 'penalty_visited': -0.5, 'penalty_no_sense_waiting': -3, 'penalty_bumping_into_each_other': -3}):
        """
        Initializes the maze environment.
        Attributes:
            grid_size (int): Len of the side of the square grid.
            render_mode (str, optional): Mode for rendering the environment.
            reward_rates (dict, optional): Reward and penalty rates for various actions and states in the environment.
            random_gate (bool): If True, the door gate position is randomized.
            switched_finish (bool): If True, the finish positions of the agents are switched.
            additional_wall (bool): If True, an additional wall is added to the maze.
        """
        self.grid_size = grid_size
        self.possible_agents = ['agent_1', 'agent_2']  # list of agents
        self.render_mode = render_mode
        self.random_gate = random_gate
        self.switched_finish = switched_finish
        self.additional_wall = additional_wall
        self.reward_rates = reward_rates
        self.reset()  # see reset function bellow

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.agents = copy(self.possible_agents)


        self.current_position = {'agent_1': [7, 0], 'agent_2': [7, 7]}
        self.wall_row = [5] # row of the wall with a door in it

        # Setting up the additional wall row if required
        self.additional_wall_row = [3] if self.additional_wall else [-1]

        # Randomizing or setting a fixed door position
        if self.random_gate:
            gate_pos = random.randint(0, 6)
            self.door_to_open = [gate_pos, gate_pos + 1]
        else:
            self.door_to_open = [3, 4]

        # Setting up the end positions of the agents
        if self.switched_finish:
            self.agent_end_cell = {'agent_1': [0, 7], 'agent_2': [0, 0]}
        else:
            self.agent_end_cell = {'agent_1': [0, 0], 'agent_2': [0, 7]}

        # Initialize rewards, cumulative rewards, terminations, truncations, and infos for each agent
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.door = 'closed'  # state of the door
        self.door_ind = [0]  # state of the door indicator

        # Initialize observations
        self.observations = {
            'agent_1': np.array(self.current_position['agent_1'] + self.current_position['agent_2'] + self.wall_row +
                                self.door_to_open + self.door_ind +
                                [self.agent_end_cell['agent_1'][1]] + self.additional_wall_row),
            'agent_2': np.array(self.current_position['agent_1'] + self.current_position['agent_2'] + self.wall_row +
                                self.door_to_open + self.door_ind +
                                [self.agent_end_cell['agent_2'][1]] + self.additional_wall_row)
        }

        # Initialize the moves counter
        self.num_moves = 0

        # Set up agent selector for alternating turns between agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Set up arrays, where the visited cells will be recorded
        self.visited = {'agent_1': np.zeros((self.grid_size, self.grid_size)),
                        'agent_2': np.zeros((self.grid_size, self.grid_size))}
        self.visited['agent_1'][self.current_position['agent_1'][0], self.current_position['agent_1'][1]] = 1
        self.visited['agent_2'][self.current_position['agent_2'][0], self.current_position['agent_2'][1]] = 1

        self.door_opening_move = 2000  # will be used when calculating at which step the door was opened during training

        # get the grid state for render
        if self.render_mode == "human":
            self.update_maze_for_render()

        return self.observations, self._cumulative_rewards, self.terminations, self.truncations, self.infos

    def step(self, action):
        """
        Executes a step in the environment for the current agent.

        Actions:
            0: move up
            1: move right
            2: move down
            3: move left
            4: wait/push
        """

        self.infos = {agent: {} for agent in self.agents}

        # Get the current agent and the next/previous one
        agent = self.agent_selection
        for name in self.possible_agents:
            if name != agent:
                other_agent = name
        self.other_agent = other_agent

        self._cumulative_rewards[self.other_agent] = 0  # reset cumulative rewards.
        # We reset it only for the previous agent because of how PPO was implemented. See PPO_agents.py
        location_backup = copy(self.current_position[agent])  # backup the current position
        wanted_step = copy(self.current_position[agent])
        # Update the wanted step based on the action
        if action == 0:
            wanted_step[0] -= 1
        elif action == 1:
            wanted_step[1] += 1
        elif action == 2:
            wanted_step[0] += 1
        elif action == 3:
            wanted_step[1] -= 1

        self.current_position[agent] = copy(wanted_step)
        # Check if the move is legal, if not - revert to the backup
        if not self.check_if_legal(wanted_step, other_agent):
            self._cumulative_rewards[agent] += self.reward_rates['penalty_illegal']
            self.infos[agent] = 'Illegal move'
            self.current_position[agent] = location_backup
            if wanted_step[0] == self.current_position[other_agent]:
                self._cumulative_rewards[agent] += self.reward_rates['penalty_bumping_into_each_other']
        elif action == 4:
            # check if two agents are in the front of the closed door and trying to open it
            cond1 = (self.current_position[agent] == [self.wall_row[0] + 1, self.door_to_open[0]] and
                     self.current_position[other_agent] == [self.wall_row[0] + 1, self.door_to_open[1]])
            cond2 = (self.current_position[agent] == [self.wall_row[0] + 1, self.door_to_open[1]] and
                     self.current_position[other_agent] == [self.wall_row[0] + 1, self.door_to_open[0]])
            if (cond1 or cond2) and self.door != 'open':
                self.infos = {agent: 'door opened!' for agent in self.agents}
                self._cumulative_rewards[agent] += self.reward_rates['reward_open_door']
                self._cumulative_rewards[other_agent] += self.reward_rates['reward_open_door']
                self.door = 'open'
                self.door_ind = [1]
                self.door_opening_move = self.num_moves
            # Encourage to not leave the end cell
            elif self.current_position[agent] == self.agent_end_cell[agent]:
                self._cumulative_rewards[agent] += 0.35
            else:
                self._cumulative_rewards[agent] += self.reward_rates['penalty_no_sense_waiting']

        # Check if the agent visited current cell before, but this cell is not the final one or near the door
        cond_for_visited1 = not (self.current_position[agent] == [self.wall_row[0] + 1, self.door_to_open[0]] or
                                 self.current_position[agent] == [self.wall_row[0] + 1, self.door_to_open[1]])
        cond_for_visited2 = not (self.current_position[agent] == self.agent_end_cell[agent])
        if (self.visited[agent][self.current_position[agent][0], self.current_position[agent][1]] == 1
                and cond_for_visited1 and cond_for_visited2):
            self._cumulative_rewards[agent] += self.reward_rates['penalty_visited']

        #  Add current cell to the list of visited cells
        self.visited[agent][self.current_position[agent][0], self.current_position[agent][1]] = 1

        #  Small constant penalty, that is reducing if agents go closer to the exit
        self._cumulative_rewards[agent] += self.reward_rates['penalty_per_move_per_row'] * self.current_position[agent][0]
        #  Penalty for the door being closed, encouraging the agents to open it as soon as possible
        if self.door == 'closed':
            self._cumulative_rewards[agent] += self.reward_rates['penalty_closed_door']
        #  Check if agents reached the exit
        if (self.current_position[agent] == self.agent_end_cell[agent] and
                self.current_position[other_agent] == self.agent_end_cell[other_agent]):
            self.infos[agent] = 'Escaped!'
            # self.infos[other_agent] = 'Escaped!'
            self._cumulative_rewards[agent] += self.reward_rates['reward_escape']
            self.terminations[agent] = True

        #  Update observations
        self.observations = {'agent_1': np.array(self.current_position['agent_1'] + self.current_position[
            'agent_2'] + self.wall_row + self.door_to_open + self.door_ind + [
                                                     self.agent_end_cell['agent_1'][1]] + self.additional_wall_row),
                             'agent_2': np.array(self.current_position['agent_1'] + self.current_position[
                                 'agent_2'] + self.wall_row + self.door_to_open + self.door_ind + [
                                                     self.agent_end_cell['agent_2'][1]] + self.additional_wall_row)}
        # Next agent
        self.agent_selection = self._agent_selector.next()
        self.num_moves += 1

        if self.render_mode == "human":
            self.update_maze_for_render()

        return self.observations, self._cumulative_rewards, self.terminations, self.truncations, self.infos

    def render(self, cell_size, screen=None):
        """
        Renders the current state of the maze environment.

        Args:
            cell_size (int): The size of each cell in the maze.
            screen (pygame.Surface, optional): Pygame surface for rendering.
        """
        if screen is None:
            return

        if not pygame.get_init():
            pygame.init()
        colors = {
            0: (255, 255, 255),  # Free cell
            1: (65, 65, 65),  # Wall
            2: (32, 178, 160),  # Current position agent 1
            3: (255, 69, 0),  # Current position agent 2
            4: (224, 255, 255),  # End position agent 1
            5: (250, 250, 210),  # End position agent 2
            6: (165, 42, 42)  # Door closed
        }

        screen.fill((200, 200, 200))  # Light grey background

        # Draw the grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = colors[self.maze_for_render[y, x]]
                pygame.draw.rect(screen, color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size), 1)

        pygame.display.flip()

    def close(self):
        """
        Closes the environment.
        """
        if self.render_mode == 'human':
            pygame.quit()
        else:
            pass

    # lru_cache allows observation and action spaces to be memorized,
    # reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Defines the observation space for each agent.

        Args:
            agent (str): The agent name, either 'agent_1' or 'agent_2'.

        Returns:
            spaces.MultiDiscrete: The observation space for the given agent.
        """
        return spaces.MultiDiscrete([8, 8, 8, 8, 8, 8, 8, 2, 8, 8])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Defines the action space for each agent.

        Args:
            agent (str): The agent name.

        Returns:
            spaces.Discrete: The action space for the given agent.
        """
        return spaces.Discrete(5)

    def check_if_legal(self, location, other_agent):
        """
        Checks if the move to the given location is legal.

        Args:
            location (list): The target location to check.
            other_agent (str): The name of the other agent.

        Returns:
            bool: True if the move is legal, False otherwise.
        """
        if location == self.current_position[other_agent]:
            return False
        elif location[0] == self.wall_row[0] and (location[1] != self.door_to_open[0] and
                                                  location[1] != self.door_to_open[1]):
            return False
        elif location[0] == self.wall_row[0] and (location[1] == self.door_to_open[0] or
                                                  location[1] == self.door_to_open[1]) and self.door == 'closed':
            return False
        elif location[0] < 0 or location[0] >= self.grid_size or location[1] < 0 or location[1] >= self.grid_size:
            return False
        elif self.additional_wall and (location[0] == self.additional_wall_row[0] and 1 <= location[1] <= 6):
            return False
        else:
            return True

    def update_maze_for_render(self):
        """
        Updates the maze state for rendering.
        """
        self.maze_for_render = np.zeros((self.grid_size, self.grid_size))

        self.maze_for_render[self.agent_end_cell['agent_1'][0], self.agent_end_cell['agent_1'][1]] = 4
        self.maze_for_render[self.agent_end_cell['agent_2'][0], self.agent_end_cell['agent_2'][1]] = 5
        self.maze_for_render[self.wall_row, :] = 1
        if self.door == 'closed':
            self.maze_for_render[self.wall_row[0], self.door_to_open[0]] = 6
            self.maze_for_render[self.wall_row[0], self.door_to_open[1]] = 6
        else:
            self.maze_for_render[self.wall_row[0], self.door_to_open[0]] = 0
            self.maze_for_render[self.wall_row[0], self.door_to_open[1]] = 0
        if self.additional_wall:
            self.maze_for_render[self.additional_wall_row[0], 1:7] = 1
        self.maze_for_render[self.current_position['agent_1'][0], self.current_position['agent_1'][1]] = 2
        self.maze_for_render[self.current_position['agent_2'][0], self.current_position['agent_2'][1]] = 3

