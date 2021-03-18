import torch
import torch.optim as optim
from utils.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer
from agents.DDQL import DDQL
from exploration_strategies.epsilon_greedy_strategy import epsilon_greedy_strategy
from neural_networks.q_learning import Convolutional_Q_Network
import os
import sys
from datetime import datetime
import pandas

class Prioritized_DDQL(DDQL):

    def __init__(self, config):
        DDQL.__init__(self, config)
        self.memory = Prioritised_Replay_Buffer(self.hyperparameters, config.seed)

    def learn(self, experiences=None):
        """ Learning step of the DDQL algorithm """

        if experiences is None:
            experiences, importance_sampling_weights = self.memory.sample()  # Sample the experiences

        states, actions, rewards, next_states, dones = experiences

        self.q_network_optimizer.zero_grad()  # Reset the gradients #

        """ First, we compute the loss """
        loss, td_errors = self.compute_loss_and_td_errors(states, actions, rewards, next_states, dones, importance_sampling_weights)

        """ Compute the gradients of the loss """
        loss.backward()

        """ Clip the loss if necessary"""
        if self.hyperparameters["gradient_clipping_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(),
                                           self.hyperparameters["gradient_clipping_norm"])

        """ Take an optimization step with the loss """
        self.q_network_optimizer.step()  # this applies the gradients

        self.memory.update_td_errors(td_errors.squeeze(1))

        return loss.cpu().detach().numpy()

    def save_experience(self, memory, experience=None):
        """Saves the latest experience including the td_error"""
        max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(max_td_error_in_experiences,*experience)

    def compute_loss_and_td_errors(self, states, actions, rewards, next_states, dones, importance_sampling_weights):
        """Calculates the loss for the local Q network. It weighs each observations loss according to the importance
        sampling weights which come from the prioritised replay buffer"""
        """ Loss computation based on the DDQL Algorithm """

        # Loss(Q_expected,Q_target)
        # Q_expected <= r + g*Q_target(s',argmax(Q(s',a')))
        # Q_target <= Q(s,a)

        """ Compute the targets values Q_target (This operation does not require gradient) """
        with torch.no_grad():
            # Compute the target (expected) Q-values
            max_action_indexes = self.q_network_local(next_states).detach().argmax(1)  # argmax(Q(s',a'))
            Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(
                1))  # Q_target(s',argmax(Q(s',a')))
            Q_targets = rewards.unsqueeze(1) + self.hyperparameters['discount_rate'] * Q_targets_next * (
                        1 - dones.unsqueeze(1))  # r + g*Q_target(s',argmax(Q(s',a')))

        Q_expected = self.q_network_local(states).gather(1, actions.long().unsqueeze(1))

        loss = self.loss_function(Q_expected, Q_targets)

        loss = loss * importance_sampling_weights

        loss = torch.mean(loss)

        td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()

        return loss, td_errors






