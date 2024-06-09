import pygame
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def evaluate_policy_visual(env, ag, delay_between_moves=100):
    """
    Visualizes the policy evaluation by rendering the environment using Pygame.

    Args:
        env: The environment to evaluate.
        ag: The agents to evaluate.
    """
    pygame.init()

    # Parameters
    cell_size = 60
    grid_size = 8
    maze_width = (grid_size) * cell_size
    maze_height = (grid_size) * cell_size
    screen_width = maze_width
    screen_height = maze_height

    # Create screen
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Escape")


    for episode in range(1, 10):
        env.reset()
        pygame.time.wait(3000)  # delay before starting the game
        done = {'agent_1': False, 'agent_2': False}
        score = {'agent_1': 0, 'agent_2': 0}
        while not all(done.values()) and env.num_moves < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            agent = env.agent_selection
            action = ag.predict(env.observations[agent], agent)
            state, reward, done, truncated, info = env.step(action)
            score[agent] += reward[agent]
            env.render(cell_size, screen)
            pygame.time.wait(delay_between_moves)

        print('Episode:{} Score:{} NumSteps:{} Door opening move:{}'.format(episode, score, env.num_moves, env.door_opening_move))
    env.close()

def evaluate_policy(env, ag, episodes, probabilistic=False):
    """
    Evaluates the policy of the agent over a given number of episodes.

    Args:
        env: The environment to evaluate.
        ag: The agents to evaluate.
        episodes (int): The number of episodes to evaluate.
        probabilistic (bool): Whether to use probabilistic action selection.
    """
    scores = {'agent_1': [], 'agent_2': []}
    moves = []
    door_opening_move = []
    for ep in range(episodes):
        env.reset()
        done = {'agent_1': False, 'agent_2': False}
        score = {'agent_1': 0, 'agent_2': 0}
        while not all(done.values()) and env.num_moves < 100:
            agent = env.agent_selection
            if probabilistic:
                action = ag.predict_probabilistic(env.observations[agent], agent)
            else:
                action = ag.predict(env.observations[agent], agent)
            state, reward, done, truncated, info = env.step(action)
            score[agent] += reward[agent]


        print('Episode:{} Score:{} NumSteps:{} Door opening move: {}'.format(ep, score, env.num_moves, env.door_opening_move))
        scores['agent_1'].append(score['agent_1'])
        scores['agent_2'].append(score['agent_2'])
        moves.append(env.num_moves)
        door_opening_move.append(env.door_opening_move)
    env.close()
    print('Mean Score:{} Mean NumSteps:{} Mean Door opening move: {}'.format({'agent_1': np.mean(scores['agent_1']),
                                                                'agent_2': np.mean(scores['agent_2'])}, np.mean(moves), np.mean(door_opening_move)))
    print('Std Score: {} Std NumSteps:{} Std Door opening move: {}'.format({'agent_1': np.std(scores['agent_1']),
                                                                'agent_2': np.std(scores['agent_2'])}, np.std(moves), np.std(door_opening_move)))


def save_model(model, save_path, model_name):
    """
    Saves the model's actor and critic networks for each agent, along with various performance metrics.

    Args:
        model: The PPO model to save.
        save_path (str): The directory path where the models and logs will be saved.
        model_name (str): The suffix to append to the model file names.
    """
    for agent in model.agents:
        # Define file paths for saving actor and critic models
        name = agent + model_name
        actor_model_path = os.path.join(save_path, f'{name}_actor1.pth')
        critic_model_path = os.path.join(save_path, f'{name}_critic1.pth')
        # Save the actor and critic model state dictionaries
        torch.save(model.actors[agent].state_dict(), actor_model_path)
        torch.save(model.critics[agent].state_dict(), critic_model_path)

    # Save the performance metrics as numpy arrays
    np.save(os.path.join(save_path, f'av_score_for_plot_ag1_{model_name}.npy'),
            np.array(model.av_score_for_plot['agent_1']))
    np.save(os.path.join(save_path, f'av_score_for_plot_ag2_{model_name}.npy'),
            np.array(model.av_score_for_plot['agent_2']))
    np.save(os.path.join(save_path, f'av_moves_for_plot_{model_name}.npy'), np.array(model.av_moves_for_plot))
    np.save(os.path.join(save_path, f'average_door_open_{model_name}.npy'), np.array(model.average_door_open))

def load_model(model, load_path, model_name):
    """
    Loads the model's actor and critic networks for each agent, along with various performance metrics.

    Args:
        model: The PPO model to load into.
        load_path (str): The directory path from where the models and logs will be loaded.
        model_name (str): The suffix appended to the model file names.
    """
    for agent in model.agents:
        # Define file paths for loading actor and critic models
        name = agent + model_name
        actor_model_path = os.path.join(load_path, f'{name}_actor1.pth')
        critic_model_path = os.path.join(load_path, f'{name}_critic1.pth')
        # Load the actor and critic model state dictionaries
        model.actors[agent].load_state_dict(torch.load(actor_model_path))
        model.critics[agent].load_state_dict(torch.load(critic_model_path))

    # Load the performance metrics as numpy arrays
    model.av_score_for_plot['agent_1'] = np.load(os.path.join(load_path, f'av_score_for_plot_ag1_{model_name}.npy')).tolist()
    model.av_score_for_plot['agent_2'] = np.load(os.path.join(load_path, f'av_score_for_plot_ag2_{model_name}.npy')).tolist()
    model.av_moves_for_plot = np.load(os.path.join(load_path, f'av_moves_for_plot_{model_name}.npy')).tolist()
    model.average_door_open = np.load(os.path.join(load_path, f'average_door_open_{model_name}.npy')).tolist()


def plot_training_results(model):
    """
    Plots the training results for the given PPO model.

    Args:
        model: The PPO model with training data to plot.
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # Plot scores for two agents
    axs[0].plot(model.av_score_for_plot['agent_1'], label='Agent 1 Score', color='b', linestyle='-', marker='o')
    axs[0].plot(model.av_score_for_plot['agent_2'], label='Agent 2 Score', color='r', linestyle='-', marker='x')
    axs[0].set_title('Scores of Agents Over Time')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Average Score')
    axs[0].legend()
    axs[0].grid(True)

    # Plot number of moves and door open counts
    axs[1].plot(model.av_moves_for_plot, label='Average Number of Moves', color='g', linestyle='-', marker='s')
    axs[1].plot(model.average_door_open, label='Average Door Open Moves', color='m', linestyle='-', marker='d')
    axs[1].set_title('Number of Moves and Door Open Counts')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Count')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout for better visualization
    plt.tight_layout()

    # Display the plots
    plt.show()