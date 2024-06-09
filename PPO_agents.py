import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from actor_critic import PPO_NN_actor, PPO_NN_critic

class PPO:
    """
    Proximal Policy Optimization (PPO) for training in a cooperative multi-agent environment.
    """
    def __init__(self, env, DEVICE, decrease_max_moves=True, print_every_episode=False, entropy_coef=0):
        """
        Initializes the PPO agent with the given environment.

        Args:
            env (gym.Env): The environment to train the agents in.
            DEVICE (torch.device): Device where the agent will be trained on.
            decrease_max_moves (bool): Defines if maximum moves before termination are decreased during training.
            print_every_episode (bools): Defines if outputs scores every iteration.
            entropy_coef (float): Coefficient for entropy regularization.
        """
        self.decrease_max_moves = decrease_max_moves
        self.print_every_episode = print_every_episode
        self.DEVICE = DEVICE
        self.n_updates_per_iteration = 5  # number of updates per training iteration.
        self.lr = 0.001  # learning rate for the optimizer.
        self.gamma = 0.95  # discount factor for reward computation.
        self.clip = 0.2  # clipping parameter for PPO.
        self.additional_moves = 1000  # parameter to control the number of moves before termination

        # environment and dimensions of observation and action spaces
        self.env = env
        self.obs_dim = self.env.observation_space('agent_1').shape[0]
        self.act_dim = self.env.action_space('agent_1').n

        self.agents = ['agent_1', 'agent_2']

        # Initialize actor and critic networks for each agent
        self.actors = {agent: PPO_NN_actor(self.obs_dim, self.act_dim).to(DEVICE) for agent in self.agents}
        self.critics = {agent: PPO_NN_critic(self.obs_dim).to(DEVICE) for agent in self.agents}

        # Initialize optimizers for actor and critic networks
        self.actor_optims = {agent: optim.Adam(self.actors[agent].parameters(), lr=self.lr) for agent in self.agents}
        self.critic_optims = {agent: optim.Adam(self.critics[agent].parameters(), lr=self.lr) for agent in self.agents}

        # Initialize memory buffers for each agent
        self.memory = {agent: deque(maxlen=10000) for agent in self.agents}
        self.batch_size = 128

        self.entropy_coef = entropy_coef

        # Initialize data structures for plotting
        self.score_for_plot = {agent: [] for agent in self.agents}
        self.av_score_for_plot = {agent: [] for agent in self.agents}
        self.moves_for_plot = []
        self.av_moves_for_plot = []
        self.average_door_open = []
        # self.actor_loss_for_plot = []
        # self.critic_loss_for_plot = []

    def learn(self, n_episodes):
        """
        Trains the agent for a given number of episodes.

        Args:
            n_episodes (int): Number of episodes to train for.
        """
        for episode in range(1, n_episodes + 1):
            # Gather data by running the agents in the environment.
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths = self.collect_trajectories(episode)

            for agent in self.agents:
                # Get the current value estimates and log probabilities
                value_estimate, _, _ = self.evaluate(batch_obs[agent], batch_acts[agent], agent)

                # Calculate the advantage estimates
                # Advantage is the difference between the actual returns and the estimated value
                advantage_estimate = batch_rtgs[agent] - value_estimate.detach()

                # Normalize advantages for better convergence, adding a small value to avoid zero in the denominator
                advantage_estimate = (advantage_estimate - advantage_estimate.mean()) / (
                            advantage_estimate.std() + 1e-10)

                # Update the networks
                for _ in range(self.n_updates_per_iteration):
                    # Evaluate the value estimates and log probabilities with the current network parameters
                    value_estimate, current_log_probs, action_probs = self.evaluate(batch_obs[agent], batch_acts[agent], agent)

                    # Calculate the probability ratios between the current policy and the old policy
                    ratios = torch.exp(current_log_probs - batch_log_probs[agent])

                    # Calculate objective function
                    objective_function = torch.min(ratios * advantage_estimate,
                                                    torch.clamp(ratios, 1 - self.clip,
                                                                1 + self.clip) * advantage_estimate)

                    # The goal is to maximize the objective function
                    actor_loss = -objective_function.mean()

                    # Add entropy regularization
                    if self.entropy_coef != 0:
                        dist = torch.distributions.Categorical(torch.clamp(action_probs, min=1e-10))
                        entropy_loss = -dist.entropy().mean()
                        actor_loss += self.entropy_coef * entropy_loss

                    # Calculate the critic loss
                    critic_loss = nn.MSELoss()(value_estimate, batch_rtgs[agent])

                    # Store the actor and critic losses for plotting
                    # self.actor_loss_for_plot.append(actor_loss.item())
                    # self.critic_loss_for_plot.append(critic_loss.item())

                    # Update the networks
                    self.actor_optims[agent].zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optims[agent].step()

                    self.critic_optims[agent].zero_grad()
                    critic_loss.backward()
                    self.critic_optims[agent].step()
    def collect_trajectories(self, episode):
        """
        Collects trajectories by running the agent in the environment.

        Args:
            episode (int): The current episode number.

        Returns:
            tuple: Batch of observations, actions, log probabilities, rewards-to-go, and lengths of episodes.
        """
        # Initialize batches
        batch_obs = {agent: [] for agent in self.agents}
        batch_acts = {agent: [] for agent in self.agents}
        batch_log_probs = {agent: [] for agent in self.agents}
        batch_rewards = {agent: [] for agent in self.agents}
        batch_rtgs = {agent: [] for agent in self.agents}
        batch_lengths = {agent: [] for agent in self.agents}

        scores = {agent: [] for agent in self.agents}
        moves = []
        door_open = []

        # Play the game several times and store data
        for k in range(self.batch_size):
            ep_rewards = {agent: [] for agent in self.agents}
            obs, _, done, truncated, _ = self.env.reset()
            score = {agent: 0 for agent in self.agents}
            while not all(done.values()) and self.env.num_moves < max(200, self.additional_moves):
                for agent in self.agents:
                    if not done[agent]:
                        action, log_prob = self.get_action(obs[agent], agent)
                        next_obs, rewards, done, _, _ = self.env.step(action)

                        batch_obs[agent].append(obs[agent])
                        batch_acts[agent].append(action)
                        batch_log_probs[agent].append(log_prob)
                        ep_rewards[agent].append(rewards[agent])

                        obs = next_obs

                        score[agent] += rewards[agent]
            if self.print_every_episode:
                print(f'Episode, iter:{episode, k} Scores:{score} NumSteps:{self.env.num_moves} Door opening move:{self.env.door_opening_move}')
            door_open.append(self.env.door_opening_move)
            scores['agent_1'].append(score['agent_1'])
            scores['agent_2'].append(score['agent_2'])
            moves.append(self.env.num_moves)

            for agent in self.agents:
                batch_lengths[agent].append(len(ep_rewards[agent]))
                batch_rewards[agent].append(ep_rewards[agent])

        for agent in self.agents:
            batch_obs[agent] = torch.tensor(np.array(batch_obs[agent]), dtype=torch.float).to(self.DEVICE)
            batch_acts[agent] = torch.tensor(np.array(batch_acts[agent]), dtype=torch.long).to(self.DEVICE)
            batch_log_probs[agent] = torch.tensor(batch_log_probs[agent], dtype=torch.float).to(self.DEVICE)
            batch_rtgs[agent] = self.compute_rewards_to_go(batch_rewards[agent])

        scores = {agent: np.mean(scores[agent]) for agent in self.agents}
        print(f'Episode:{episode} Average Score:{scores} Average NumSteps:{np.mean(moves)} Average door open {np.mean(door_open)}')
        self.av_score_for_plot['agent_1'].append(scores['agent_1'])
        self.av_score_for_plot['agent_2'].append(scores['agent_2'])
        self.av_moves_for_plot.append(np.mean(moves))
        self.average_door_open.append(np.mean(door_open))
        if self.decrease_max_moves:
            self.additional_moves -= 50

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths

    def compute_rewards_to_go(self, batch_rewards):
        """
        Computes rewards-to-go for each episode.

        Args:
            batch_rewards (list): List of rewards for each episode.

        Returns:
            torch.Tensor: Rewards-to-go.
        """
        batch_rtgs = []
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float).to(self.DEVICE)
        return batch_rtgs

    def get_action(self, obs, agent):
        """
        Samples an action from the policy distribution for the given observation.

        Args:
            obs (np.ndarray): The observation from the environment.
            agent (str): The agent name.

        Returns:
            tuple: The action and its log probability.
        """
        state = torch.tensor(obs, dtype=torch.float).to(self.DEVICE)
        action_probs = self.actors[agent](state).to(self.DEVICE)
        action_probs = torch.clamp(action_probs, min=1e-10)
        dist = torch.distributions.Categorical(action_probs)
        other_agent = next(name for name in self.agents if name != agent)
        if self.env.terminations[other_agent]:
            action = torch.tensor([4])
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze().item(), log_prob.squeeze().detach()

    def evaluate(self, batch_obs, batch_acts, agent):
        """
        Evaluates the value and log probabilities of the given batch of observations and actions.

        Args:
            batch_obs (torch.Tensor): Batch of observations.
            batch_acts (torch.Tensor): Batch of actions.
            agent (str): The agent name.

        Returns:
            tuple: The value estimates and log probabilities.
        """
        value_estimate = self.critics[agent](batch_obs).squeeze()
        action_probs = self.actors[agent](batch_obs)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(batch_acts)
        return value_estimate, log_probs, action_probs
    def predict(self, obs, agent):
        """
        Predicts the best action for the given observation.

        Args:
            obs (np.ndarray): The observation from the environment.
            agent (str): The agent name.

        Returns:
            int: The predicted action.
        """
        state = torch.tensor(obs, dtype=torch.float).to(self.DEVICE)
        action_probs = self.actors[agent](state).to(self.DEVICE)
        action = torch.argmax(action_probs, dim=-1).item()
        return action
    def predict_probabilistic(self, obs, agent):
        """
        Sample action from the probability distribution prediction.

        Args:
            obs (np.ndarray): The observation from the environment.
            agent (str): The agent name.

        Returns:
            int: The predicted action.
        """
        state = torch.tensor(obs, dtype=torch.float).to(self.DEVICE)
        action_probs = self.actors[agent](state).to(self.DEVICE)
        action_probs = torch.clamp(action_probs, min=1e-10)
        dist = torch.distributions.Categorical(action_probs)
        other_agent = next(name for name in self.agents if name != agent)
        if self.env.terminations[other_agent]:
            action = torch.tensor([4])
        else:
            action = dist.sample()
        return action