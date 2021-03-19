import torch
from utils.Replay_buffer import Replay_Buffer
from neural_networks.discrete_soft_actor_critic import Convolutional_CriticNetwork, Convolutional_ActorNetwork
from exploration_strategies.OUNoise import OrnsteinUhlenbeckActionNoise
from agents.Soft_AC import Soft_Actors_Critic
from torch.distributions import Categorical
import os
import sys
from datetime import datetime
import pandas
import numpy as np

class Discrete_Soft_Actor_Critic(Soft_Actors_Critic):
    """ Inherits the Continuous SAC class and implements
    some changes in order to adapt to discrete action size. """

    def __init__(self, config):

        self.dirname = datetime.now().strftime(sys.path[0]+"\RESULTS\Discrete_SAC_%d-%m-%Y_%H_%M_%S").replace('\\','/')

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

        # Buffer memory #
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"],
                                    self.hyperparameters["batch_size"],
                                    self.seed,
                                    self.device)

        # Creation of local critics Q1(s,a) and Q2(s,a) like in Twin Delayed Det. Deep Policy Gradient #
        self.critic_local_1 = self.create_critic_cnn(config.state_size, config.environment.action_size, self.device)
        self.critic_local_2 = self.create_critic_cnn(config.state_size, config.environment.action_size, self.device)

        # Creation of the critic targets #
        self.critic_target_1 = self.create_critic_cnn(config.state_size, config.environment.action_size, self.device)
        self.critic_target_2 = self.create_critic_cnn(config.state_size, config.environment.action_size, self.device)

        # Copy the local into the target #
        self.copy_model(self.critic_local_1, self.critic_target_1)
        self.copy_model(self.critic_local_2, self.critic_target_2)

        # Create the actor network #
        self.actor_local = self.create_actor_cnn(config.state_size, config.environment.action_size, self.device)

        # Create the optimizers for the different neural networks #
        self.optimizer_critic_1 = torch.optim.Adam(self.critic_local_1.parameters(),
                                                   lr=self.hyperparameters['Critic']['learning_rate'])
        self.optimizer_critic_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters['Critic']['learning_rate'])
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters['Actor']['learning_rate'])

        # Entropic target parameters - Initially with the heuristic value from the original paper: a <--dim(A)#
        self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"])

        # Noise generator - Behavioral explorative policy #
        self.explorative_noise = OrnsteinUhlenbeckActionNoise(mu=self.hyperparameters['mu'],
                                                              sigma=self.hyperparameters['sigma'],
                                                              theta=self.hyperparameters['theta'],
                                                              dt=self.hyperparameters['dt'],
                                                              seed=self.seed)

        self.reset()

    def produce_action_with_actor(self, state):

        """ Produce an action using the actor network, the log probability and the tanh of the mean action """
        actor_decision = self.actor_local(state)  # decission <= [[pa1 pa2 pa3 ...]]

        # Select the most probable action #
        action_with_max_probability = torch.argmax(actor_decision, dim=1)

        # Create a categorical probability distribution with the actor decission #
        action_distribution = Categorical(actor_decision)
        # Sample an action from it #
        action = action_distribution.sample().cpu()

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = actor_decision == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(actor_decision + z)

        return action, (actor_decision, log_action_probabilities), action_with_max_probability

    def compute_critic_losses(self, states, actions, rewards, next_states, dones):

        with torch.no_grad():

            # Produce an action based on the actor #
            next_state_actions, (actor_decision, log_action_probabilities), _ = self.produce_action_with_actor(next_states)

            # Compute the target for both critic networks #
            qf1_next_target = self.critic_target_1(next_states)
            qf2_next_target = self.critic_target_2(next_states)

            # Like in T3DPG only choose the minimum of both adding the alpha value #

            min_q_next_target =  actor_decision * (torch.min(qf1_next_target, qf2_next_target)
                                                   - self.alpha * log_action_probabilities)

            min_q_next_target = min_q_next_target.sum(dim=1).unsqueeze(-1)

            # Compute the Q_target(s',a) #
            q_target = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.hyperparameters['discount_rate'] * min_q_next_target

        # Compute the local critics #
        q_local_1 = self.critic_local_1(states).gather(1, actions.long().unsqueeze(1))
        q_local_2 = self.critic_local_2(states).gather(1, actions.long().unsqueeze(1))

        # Compute the loss function #
        qf1_loss = torch.nn.functional.mse_loss(q_local_1, q_target)
        qf2_loss = torch.nn.functional.mse_loss(q_local_2, q_target)

        return qf1_loss, qf2_loss

    def compute_actor_loss(self, states):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_with_actor(states)

        qf1_pi = self.critic_local_1(states)
        qf2_pi = self.critic_local_2(states)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        inside_term = self.alpha * log_action_probabilities - min_qf_pi

        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()

        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)

        return policy_loss, log_action_probabilities

    @staticmethod
    def create_critic_cnn(input_dim, output_dim, device):

        neural_network = Convolutional_CriticNetwork(input_dim, output_dim).to(device)

        return neural_network

    @staticmethod
    def create_actor_cnn(input_dim, output_dim, device):

        neural_network = Convolutional_ActorNetwork(input_dim, output_dim).to(device)

        return neural_network

    def perform_action(self, action):

        self.next_state, self.reward, self.done, _ = self.environment.step(action)

        self.episodic_reward += self.reward


