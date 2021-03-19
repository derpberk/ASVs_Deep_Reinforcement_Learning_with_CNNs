"""

Double Deep Q Learning Agent

"""

import torch
import torch.optim as optim
from utils.Replay_buffer import Replay_Buffer
from exploration_strategies.epsilon_greedy_strategy import epsilon_greedy_strategy
from neural_networks.q_learning import Convolutional_Q_Network
import os
import sys
from datetime import datetime
import pandas

class DDQL(object):

    def __init__(self, config):

        self.dirname = datetime.now().strftime(sys.path[0]+"\RESULTS\DDQL_%d-%m-%Y_%H_%M_%S").replace('\\' ,'/')

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

        # Creation of the behavioral network #
        self.q_network_local = self.create_cnn(input_dim=config.state_size,
                                               output_dim=config.action_size,
                                               device=self.device)

        # Creation of the target network #
        self.q_network_target = self.create_cnn(input_dim=config.state_size,
                                                output_dim=config.action_size,
                                                device=self.device)

        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"])

        # Copy local into the target #
        self.copy_model(source=self.q_network_local, destiny=self.q_network_target)

        # Exploratory strategy #
        self.exploratory_strategy = epsilon_greedy_strategy(initial_epsilon=self.hyperparameters['initial_epsilon'],
                                                            decremental_epsilon=self.hyperparameters['epsilon_decrement'],
                                                            epsilon_min=self.hyperparameters['epsilon_min'],
                                                            seed=self.seed)

        self.loss_function = torch.nn.SmoothL1Loss() # Huber - Loss Function #

        self.reset()

    @staticmethod
    def copy_model(source, destiny):
        """ Copy the parameters of a network into another """
        for source, destiny in zip(source.parameters(), destiny.parameters()):
            destiny.data.copy_(source.data.clone())

    def training_step(self) -> dict:
        """ Execution of one step of the learning process """

        self.state = self.environment.reset()
        self.episodic_reward = 0
        self.episodic_loss = 0
        self.done = False

        while not self.done:

            # Choose the action based on the policy #
            self.action = self.pick_action()

            # Perform the action #
            self.perform_action(self.action)

            # Perform a learning step #
            if self.enough_experiences():
                loss = self.learn()
                self.episodic_loss += loss

            # Save the experience #
            self.save_experience(self.memory)

            # Update the state #
            self.state = self.next_state

        # Update the epsilon
        self.exploratory_strategy.update_parameters()

        # Update the target function (if it is time) #
        self.update_target_function()

        self.episode_number += 1
        self.update_rolling_metrics(0.05)

        return self.report_progress()

    def reset(self):

        """ Reset the Agent atributes """
        self.state = []
        self.next_state = []
        self.reward = []
        self.done = False
        self.action = []
        self.episode_number = 0
        self.episodic_reward = 0
        self.episodic_reward_buffer = []
        self.rolling_reward = []
        self.episodic_loss = 0
        self.rolling_loss = []
        self.record = float('-inf')
        self.exploratory_strategy.reset()
        self.episodic_record = []

    def pick_action(self, state=None):

        """ Pick the action depending on the exploration policy """

        if state is None:
            state = self.state

        # Local network to evaluation mode #
        self.q_network_local.eval()

        state = torch.tensor([state],device=self.device).float()

        with torch.no_grad():
            action_values = self.q_network_local(state).cpu().detach().numpy().squeeze(0) # Evaluate the Q-function

        # Return to training mode
        self.q_network_local.train()

        selected_action = self.exploratory_strategy.pick_action(action_values)

        return selected_action

    def perform_action(self, action):

        self.next_state, self.reward, self.done, _ = self.environment.step(action)

        self.episodic_reward += self.reward

    def enough_experiences(self):
        """ Check if there are enough experiences in the buffer to sample a batch """
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory, experience=None):
        """ Save the experience in the buffer replay """

        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        # Store transition !
        memory.add_experience(*experience)

    def learn(self, experiences=None):
        """ Learning step of the DDQL algorithm """

        if experiences is None:
            experiences = self.memory.sample() # Sample the experiences

        states, actions, rewards, next_states, dones = experiences

        self.q_network_optimizer.zero_grad()  # Reset the gradients #

        """ First, we compute the loss """
        loss = self.compute_loss(states, actions, rewards, next_states, dones)

        """ Compute the gradients of the loss """
        loss.backward()

        """ Clip the loss if necessary"""
        if self.hyperparameters["gradient_clipping_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), self.hyperparameters["gradient_clipping_norm"])

        """ Take an optimization step with the loss """
        self.q_network_optimizer.step()  # this applies the gradients

        return loss.cpu().detach().numpy()

    def compute_loss(self, states, actions, rewards, next_states, dones):
        """ Loss computation based on the DDQL Algorithm """

        # Loss(Q_expected,Q_target)
        # Q_expected <= r + g*Q_target(s',argmax(Q(s',a')))
        # Q_target <= Q(s,a)

        """ Compute the targets values Q_target (This operation does not require gradient) """
        with torch.no_grad():
            # Compute the target (expected) Q-values
            max_action_indexes = self.q_network_local(next_states).detach().argmax(1) # argmax(Q(s',a'))
            Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1)) # Q_target(s',argmax(Q(s',a')))
            Q_targets = rewards.unsqueeze(1) + self.hyperparameters['discount_rate'] * Q_targets_next * (1 - dones.unsqueeze(1))  #r + g*Q_target(s',argmax(Q(s',a')))

        Q_expected = self.q_network_local(states).gather(1, actions.long().unsqueeze(1))

        loss = self.loss_function(Q_expected, Q_targets)

        return loss

    @staticmethod
    def create_cnn(input_dim, output_dim, device):

        neural_network = Convolutional_Q_Network(input_dim, output_dim).to(device)

        return neural_network

    def update_target_function(self):
        if self.episode_number % self.hyperparameters["target_update_freq"] == 0:
            self.hard_update_of_target_network(self.q_network_local, self.q_network_target)  # Hard target update

    @staticmethod
    def hard_update_of_target_network(local_model, target_model):
        """Updates the target network in the direction of the local network by hard substitution"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def report_progress(self):

        progress = {}
        progress['reward'] = self.episodic_reward
        progress['loss'] = self.rolling_loss[-1]
        progress['rolling_reward'] = self.rolling_reward[-1]
        progress['epsilon'] = self.exploratory_strategy.epsilon
        progress['episode_number'] = self.episode_number
        progress['number_of_episodes'] = self.number_of_episodes
        progress['record'] = self.record

        return progress

    def update_rolling_metrics(self, tau):

        if self.episodic_reward > self.record:
            self.record = self.episodic_reward

        self.episodic_reward_buffer.append(self.episodic_reward)

        if self.rolling_reward == []:
            self.rolling_reward.append(self.episodic_reward)
        else:
            self.rolling_reward.append(self.rolling_reward[-1]*(1-tau) + self.episodic_reward*tau)

        if self.rolling_loss == []:
            self.rolling_loss.append(self.episodic_loss)
        else:
            self.rolling_loss.append(self.rolling_loss[-1]*(1-tau) + self.episodic_loss*tau)

        self.episodic_record.append(self.record)

    def save_progress(self, save_models = False):
        """ Function to save the progress of the training """

        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)

        data = {}
        data['Episodic Reward'] = self.episodic_reward
        data['Filtered Episodic Reward'] = self.rolling_reward
        data['Record'] = self.episodic_record
        data['Average Episodic Loss'] = self.rolling_loss

        db = pandas.DataFrame(data)

        db.to_csv(self.dirname + '/experiment_results.csv', index_label = 'Episodes')

        if save_models:
            torch.save(self.q_network_local, self.dirname + '\q_network_local_EPISODE_{}'.format(self.episode_number))












