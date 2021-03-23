import random
import torch
import sys
from contextlib import closing
from torch.multiprocessing import Pool
from random import randint
from exploration_strategies.OUNoise import OrnsteinUhlenbeckActionNoise


class Parallel_Experience_Generator(object):
    """ Plays n episode in parallel using a fixed agent. """

    def __init__(self, environment, policy, seed, hyperparameters, action_size, use_GPU=False, action_choice_output_columns=None):
        self.use_GPU = use_GPU
        self.environment = environment
        self.policy = policy
        self.action_choice_output_columns = action_choice_output_columns
        self.hyperparameters = hyperparameters
        self.noise = OrnsteinUhlenbeckActionNoise(mu=[0 for _ in range(self.environment.action_shape[1])],
                                                  sigma=0.15,
                                                  theta=.01,
                                                  dt=1e-2,
                                                  seed=seed)

    def play_n_episodes(self, n):
        """Plays n episodes in parallel using the fixed policy and returns the data"""

        with closing(Pool(processes=n)) as pool:
            results = pool.map(self, range(n))
            pool.terminate()

        states_for_all_episodes = [episode[0] for episode in results]
        actions_for_all_episodes = [episode[1] for episode in results]
        rewards_for_all_episodes = [episode[2] for episode in results]

        return states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes

    def play_1_episode(self, epsilon_exploration):
        """Plays 1 episode using the fixed policy and returns the data"""

        state = self.reset_game()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        while not done:
            action = self.pick_action(self.policy, state)
            next_state, reward, done, _ = self.environment.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
        return episode_states, episode_actions, episode_rewards

    def reset_game(self):
        """Resets the game environment so it is ready to play a new episode"""
        seed = randint(0, sys.maxsize)
        torch.manual_seed(seed) # Need to do this otherwise each worker generates same experience
        state = self.environment.reset()
        return state

    def pick_action(self, policy, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        actor_output = policy(state)

        if self.action_choice_output_columns is not None:
            actor_output = actor_output[:, self.action_choice_output_columns]

        action_distribution = self.create_distributions(policy, self.environment.action_size)
        action = action_distribution.sample().cpu()

        action += torch.Tensor(self.noise())

        return action.detach().numpy()

    @staticmethod
    def create_distributions(policy_output, number_of_actions):

        means = policy_output[:, :number_of_actions].squeeze(0)
        stds = policy_output[:, number_of_actions:].squeeze(0)

        action_distribution = torch.distributions.normal.Normal(means.squeeze(0), torch.abs(stds))

        return action_distribution