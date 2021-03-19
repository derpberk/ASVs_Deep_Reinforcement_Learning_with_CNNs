import torch
from utils.Replay_buffer import Replay_Buffer
from neural_networks.soft_actor_critic import Convolutional_CriticNetwork, Convolutional_ActorNetwork
from exploration_strategies.OUNoise import OrnsteinUhlenbeckActionNoise
import numpy as np
import os
import sys
from datetime import datetime
import pandas

class Soft_Actors_Critic(object):

    def __init__(self, config):

        self.dirname = datetime.now().strftime(sys.path[0]+"\RESULTS\SAC_%d-%m-%Y_%H_%M_%S").replace('\\','/')

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

    def reset(self):

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
        self.explorative_noise.reset()
        self.number_of_steps = 0
        self.episodic_record = []

    def training_step(self):
        """ Run one episode of the enviornment """
        is_eval_episode = self.episode_number % self.hyperparameters['training_episodes_per_eval_episodes'] == 0

        if is_eval_episode: print('Evaluation episode!')

        self.state = self.environment.reset()
        self.episodic_reward = 0
        self.episodic_loss = 0
        self.done = False

        while not self.done:

            # Selected action#
            self.action = self.pick_action(is_eval_episode)

            # Perform the selected action #
            self.perform_action(self.action)

            # Perform a learning step #
            if self.enough_experiences() and self.enough_initial_steps():
                loss = self.learn()
                self.episodic_loss += loss

            # Save the experience if we are not in a eval episode #
            if not is_eval_episode:
                self.save_experience(memory=self.memory,
                                     experience=(self.state, self.action, self.reward, self.next_state, self.done))

            # Update the state #
            self.state = self.next_state

            self.number_of_steps += 1

        self.episode_number += 1

        self.update_rolling_metrics(0.05)

        return self.report_progress()

    def pick_action(self, is_eval_episode = False, state = None):
        """ Choose an action randomly during a certain number of initial steps or using the actor in training/eval mode.
            In training mode the actor is more exploratory because of alpha. """

        if state is None:
            state = self.state

        if self.enough_initial_steps() is False: # Random action selection #
            action = self.environment.action_space.sample()
        elif is_eval_episode is False: # Use actor in training mode #
            state = torch.FloatTensor([state]).to(self.device)
            action, _, _ = self.produce_action_with_actor(state)
            action = action[0].detach().cpu().numpy()
        else:
            with torch.no_grad():
                state = torch.FloatTensor([state]).to(self.device)
                _, z, action = self.produce_action_with_actor(state)
            action = action[0].detach().cpu().numpy()

        # Add noise #
        if self.hyperparameters['add_extra_noise']:
            action += self.explorative_noise()

        return action

    @staticmethod
    def create_critic_cnn(input_dim, output_dim, device):

        neural_network = Convolutional_CriticNetwork(input_dim, output_dim).to(device)

        return neural_network

    @staticmethod
    def create_actor_cnn(input_dim, output_dim, device):

        neural_network = Convolutional_ActorNetwork(input_dim, output_dim).to(device)

        return neural_network

    @staticmethod
    def copy_model(source, destiny):
        """ Copy the parameters of a network into another """
        for source, destiny in zip(source.parameters(), destiny.parameters()):
            destiny.data.copy_(source.data.clone())

    def enough_experiences(self):
        """ Check if there are enough experiences in the buffer to sample a batch """
        return len(self.memory) > self.hyperparameters["batch_size"]

    def enough_initial_steps(self):

        if self.number_of_steps < self.hyperparameters['initial_random_steps_number']:
            return False
        else:
            return True

    def save_experience(self, memory, experience=None):
        """ Save the experience in the buffer replay """

        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        # Store transition !
        memory.add_experience(*experience)

    def produce_action_with_actor(self,state):

        """ Produce an action using the actor network, the log probability and the tanh of the mean action """
        actor_decision = self.actor_local(state) # decission <= [[ mean(A1) mean(A2) ... | std(A2) std(A2) ...]]
        mean = actor_decision[:, :self.environment.action_size]  # MU
        log_std = actor_decision[:, self.environment.action_size:]  # STD
        std = log_std.exp()

        # Creation of a Normal distribution #
        normal_distribution = torch.distributions.Normal(mean, std)

        # Sample the unbounded action using the reparametrization trick #
        x = normal_distribution.rsample()
        action = torch.tanh(x)

        # Compute the log_prob for training #
        log_prob = normal_distribution.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + self.hyperparameters['EPSILON'])
        log_prob = log_prob.sum(1, keepdim = True)

        return action, log_prob, torch.tanh(mean)

    def perform_action(self, action):

        if action.shape[0] == 1: action = action[0]

        self.next_state, self.reward, self.done, _ = self.environment.step(action)

        self.episodic_reward += self.reward

    def learn(self, experiences=None):
        """ Learning process for the actor and critics """

        if experiences is None:
            experiences = self.memory.sample()  # Sample the experiences

        states, actions, rewards, next_states, dones = experiences

        # -- Optimization fo the critic -- #

        # First, we calculate the critic losses #

        q1_loss, q2_loss = self.compute_critic_losses(states, actions, rewards, next_states, dones)

        # Take an optimization step for critic optimizers #
        self.optimizer_critic_1.zero_grad()
        q1_loss.backward()
        self.optimizer_critic_1.step()  # this applies the gradients for the critic 1

        self.optimizer_critic_2.zero_grad()
        q2_loss.backward()
        self.optimizer_critic_2.step()  # this applies the gradients for the critic 2

        # Updating of the parameters #
        if self.hyperparameters['target_update_mode'] == 'soft':
            self.soft_update_target_functions()
        elif self.hyperparameters['target_update_mode'] == 'hard':
            if self.episode_number % self.hyperparameters['hard_update_frequency'] == 0:
                self.hard_update_target_functions()
        else:
            exit('BAD UPDATE STRATEGY: Choose soft/hard')

        # -- Optimization for the actor -- #

        policy_loss, log_pi = self.compute_actor_loss(states)
        alpha_loss = self.compute_entropy_tunning_loss(log_pi)

        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()

        """
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()
        """

        self.alpha = alpha_loss.exp()

        avg_critic_loss = np.mean((q1_loss.cpu().detach().numpy(),
                                   q2_loss.cpu().detach().numpy()))

        return avg_critic_loss

    def hard_update_target_functions(self):
        for target_param, local_param in zip(self.critic_target_1.parameters(), self.critic_local_1.parameters()):
            target_param.data.copy_(local_param.data)

        for target_param, local_param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_update_target_functions(self):

        for target_param, local_param in zip(self.critic_target_1.parameters(), self.critic_local_1.parameters()):
            target_param.data.copy_(target_param.data * (1-self.hyperparameters['tau']) + local_param.data * self.hyperparameters['tau'])

        for target_param, local_param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
            target_param.data.copy_(target_param.data * (1-self.hyperparameters['tau']) + local_param.data * self.hyperparameters['tau'])

    def compute_critic_losses(self, states, actions, rewards, next_states, dones):

        with torch.no_grad():

            # Produce an action based on the actor #
            next_state_action, next_state_log_pi, _ = self.produce_action_with_actor(next_states)

            # Compute the target for both critic networks #
            qf1_next_target = self.critic_target_1(next_states, actions)
            qf2_next_target = self.critic_target_2(next_states, actions)

            # Like in T3DPG only choose the minimum of both adding the alpha value #

            min_q_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

            # Compute the Q_target(s',a) #
            q_target = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.hyperparameters['discount_rate'] * min_q_next_target

        # Compute the local critics #
        q_local_1 = self.critic_local_1(states, actions)
        q_local_2 = self.critic_local_2(states, actions)

        # Compute the loss function #
        qf1_loss = torch.nn.functional.mse_loss(q_local_1, q_target)
        qf2_loss = torch.nn.functional.mse_loss(q_local_2, q_target)

        return qf1_loss, qf2_loss

    def compute_actor_loss(self, states):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""

        # Process the actor decision #
        action, log_pi, _ = self.produce_action_with_actor(states)

        # Compute the critic opinion #
        qf1_pi = self.critic_local_1(states, action)
        qf2_pi = self.critic_local_2(states, action)

        # Select the lowest #
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Compute the policy loss - as in every policy it. but pondering with alpha
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        return policy_loss, log_pi

    def compute_entropy_tunning_loss(self, log_pi):
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        return alpha_loss

    def update_rolling_metrics(self, tau):

        if self.episodic_reward > self.record:
            self.record = self.episodic_reward

        self.episodic_reward_buffer.append(self.episodic_reward)

        if not self.rolling_reward:
            self.rolling_reward.append(self.episodic_reward)
        else:
            self.rolling_reward.append(self.rolling_reward[-1]*(1-tau) + self.episodic_reward*tau)

        if not self.rolling_loss:
            self.rolling_loss.append(self.episodic_loss)
        else:
            self.rolling_loss.append(self.rolling_loss[-1]*(1-tau) + self.episodic_loss*tau)

        self.episodic_record.append(self.record)

    def report_progress(self):

        progress = {}
        progress['reward'] = self.episodic_reward
        progress['loss'] = self.rolling_loss[-1]
        progress['rolling_reward'] = self.rolling_reward[-1]
        progress['epsilon'] = self.alpha.item()
        progress['episode_number'] = self.episode_number
        progress['number_of_episodes'] = self.number_of_episodes
        progress['record'] = self.record

        return progress


    def save_progress(self, save_models = False):
        """ Function to save the progress of the training """

        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)

        data = {}
        data['Episodic Reward'] = self.episodic_reward_buffer
        data['Filtered Episodic Reward'] = self.rolling_reward
        data['Record'] = self.episodic_record
        data['Average Episodic Loss'] = self.rolling_loss

        db = pandas.DataFrame(data)

        db.to_csv(self.dirname + '/experiment_results.csv', index_label = 'Episodes')

        if save_models:
            torch.save(self.actor_local, self.dirname + r'\actor_network_local_EPISODE_{}'.format(self.episode_number))






