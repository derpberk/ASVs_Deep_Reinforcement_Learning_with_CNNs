import torch
from neural_networks.ppo import Actor_Network
from datetime import datetime
from utils.Paralell_Experience_Generator import Parallel_Experience_Generator
import sys, os
import numpy as np

class PPO(object):

    """ Proximal Policy Optimization Algorithm with paralell experience generator. """

    def __init__(self, config):
        self.dirname = datetime.now().strftime(sys.path[0] + "\RESULTS\PPO_%d-%m-%Y_%H_%M_%S").replace('\\', '/')

        # Set the seed #
        self.seed = config.seed
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.seed)

        # Get the hyperparameters #
        self.hyperparameters = config.hyperparameters

        # Set the environment #
        self.environment = config.environment

        # Number of episodes #
        self.number_of_episodes = config.number_of_episodes

        # Set the device #
        self.device = config.device

        # Instantiate the new/old policy PI #

        self.policy_new = self.create_actor_cnn()
        self.policy_old = self.create_actor_cnn()

        # Clone the policies #
        self.copy_model(self.policy_new, self.policy_old)

        # Instantiate the optimizer #
        self.policy_optimizer = self.policy_new_optimizer = torch.optim.Adam(self.policy_new.parameters(),
                                                                             lr=self.hyperparameters["learning_rate"])

        # Paralell Experience Generator #
        self.experience_generator = Parallel_Experience_Generator(self.environment,
                                                                  self.policy_new,
                                                                  self.seed,
                                                                  self.hyperparameters,
                                                                  self.environment.action_size)

        self.reset()


    def reset(self):
        """ Reset the metrics and buffers. """

        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.episode_number = []

        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.done = False
        self.action = []
        self.episode_number = 0
        self.episodic_reward = 0
        self.episodic_reward_buffer = []
        self.rolling_reward = []
        self.episodic_loss = 0
        self.rolling_loss = []
        self.record = float('-inf')
        self.number_of_steps = 0
        self.episodic_record = []

    def step(self):
        """ Run one/N episode of the environment and update the policy. """

        # Play N episodes #
        self.states_buffer, self.actions_buffer, self.rewards_buffer = \
            self.experience_generator.play_n_episodes(self.hyperparameters["episodes_per_learning_round"])
        # Accumulate N #
        self.episode_number += self.hyperparameters["episodes_per_learning_round"]

        # Update the policy according to the loss #
        self.policy_learn()

        # Equals the old and the new policy #
        self.update_old_policy()

        # Update the epsilon in the e-greedy exploratory strategy #
        self.exploratory_strategy.update_parameters()

    def policy_learn(self):
        """ Gradient descent step over the loss function. """

        # First, we compute the discounted rewards #
        discounted_returns = self.compute_discounted_returns()

        # Normalize the rewards #
        mean_rewards = np.mean(discounted_returns)
        std_rewards = np.std(discounted_returns)
        normalized_discounted_returns = (discounted_returns-mean_rewards) / (std_rewards + 1E-8)

        for _ in range(self.hyperparameters['learning_iterations_per_round']):

            ratio_of_policy_probabilities = self.compute_ratios()
            loss = self.compute_actor_loss([ratio_of_policy_probabilities], normalized_discounted_returns)

            # Take an optimization step for the actor policy #
            self.policy_optimizer.zero_grad()  # Reset the gradient
            loss.backward()  # Compute the gradient
            torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(),
                                           self.hyperparameters["gradient_clipping_norm"])  # Clip the gradient
            self.policy_optimizer.step()  # Update the parameters

    def compute_actor_loss(self, ratios_of_probs, discounted_returns):
        """ Compute the loss of the PPO """

        #TODO: En esta variante, la funcion de Advantage viene a ser directamente los rewards to go, como en el caso del REINFORCE.
        #TODO: Una forma mejor, quizás sea la implementación de una red de target al estilo del actors critic.

        # Squeeze the tensor and clamp the value to the highest number possible #
        ratios_of_probs = torch.squeeze(torch.stack(ratios_of_probs))
        ratios_of_probs = torch.clamp(ratios_of_probs, -sys.maxsize, sys.maxsize)

        discounted_returns = torch.tensor(discounted_returns).to(ratios_of_probs)

        # Compute the unclipped and clipped losses according to the paper
        potential_loss_value_1 = discounted_returns * ratios_of_probs
        potential_loss_value_2 = discounted_returns * torch.clamp(ratios_of_probs,
                                                                  min=1.0 - self.hyperparameters["clip_epsilon"],
                                                                  max=1.0 + self.hyperparameters["clip_epsilon"])
        # Select the lower value ( pessimistic choice )
        loss = torch.min(potential_loss_value_1, potential_loss_value_2)

        loss = -torch.mean(loss)

        return loss

        
    def compute_ratios(self):
        """ ratio = Pi_new(s,a) / Pi_old(s,a) -> The ratio measure the probability of the new policy to choose the given
        action of the old policy. """

        # Unpack the states/actions and convert to Tensor #
        states_batch = [state for states in self.states_buffer for state in states]
        states_batch = torch.stack([torch.Tensor(states).float().to(self.device) for states in states_batch])

        actions_batch = [action for actions in self.actions_buffer for action in actions]
        actions_batch = torch.stack([torch.Tensor(actions).float().to(self.device) for actions in actions_batch])

        # Unsqueeze the action batch #
        actions_batch = actions_batch.view(-1, len(actions_batch))

        # Evaluate both old and new policies #
        new_policy_distribution = self.compute_log_prob_of_actions(self.policy_new, states_batch, actions_batch)
        old_policy_distribution = self.compute_log_prob_of_actions(self.policy_old, states_batch, actions_batch)

        # ratio = Pi_new(s,a) / Pi_old(s,a) #
        ratios = torch.exp(new_policy_distribution) / (torch.exp(old_policy_distribution) + 1e-8)

        return ratios

    def compute_log_prob_of_actions(self, policy, states, actions):

        # Evaluate the actor, which will output a mean and std for every action dimension #
        policy_output = policy(states)
        policy_distribution = self.create_distributions(policy_output, self.environment.action_size)

        return policy_distribution.log_prob(actions)

    @staticmethod
    def create_distributions(policy_output, number_of_actions):

        means = policy_output[:, :number_of_actions].squeeze(0)
        stds = policy_output[:, number_of_actions:].squeeze(0)

        action_distribution = torch.distributions.normal.Normal(means.squeeze(0), torch.abs(stds))

        return action_distribution

    def update_old_policy(self):
        """ pi_old(s) <-- pi_new(s) """

        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
            old_param.data.copy_(new_param.data)


    @staticmethod
    def create_actor_cnn(input_dim, output_dim, device):

        neural_network = Actor_Network(input_dim, output_dim).to(device)

        return neural_network

    @staticmethod
    def copy_model(source, destiny):
        """ Copy the parameters of a network into another """
        for source, destiny in zip(source.parameters(), destiny.parameters()):
            destiny.data.copy_(source.data.clone())

    def compute_discounted_returns(self):
        """ Compute the so-called rewards to-go """

        discounted_returns_buffer = []

        for e in range(len(self.states_buffer)): # Iterate over the played episodes #

            discounted_returns = [0]

            for step in range(len(self.states_buffer[e])): # Iterate over the reward of every episode
                # R(t+1) <- R(t) + AccR(0,t) * gamma
                reward = self.rewards_buffer[e][-(step + 1)] + self.hyperparameters["discount_rate"] * discounted_returns[-1]
                discounted_returns.append(reward)
            
            discounted_returns = discounted_returns[1:]  # Discard the initial 0
            discounted_returns_buffer.extend(discounted_returns[::-1]) # Append the discounted reward to the buffer

        return discounted_returns_buffer


            


