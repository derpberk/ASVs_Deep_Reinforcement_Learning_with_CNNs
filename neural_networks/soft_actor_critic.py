from abc import ABC
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Convolutional_ActorNetwork(nn.Module, ABC):
    """
    Convolutional Neural Network for the actor.
    The Output corresponds with a Softmax layer representing
    the probability of select an action a -> P(a|s) = pi(s,action = a)

    """

    def __init__(self, input_size, action_size):
        super(Convolutional_ActorNetwork, self).__init__()

        """ Convolutional DNN """

        self.conv1 = nn.Conv2d(input_size[0], 16, 5)
        self.conv2 = nn.Conv2d(16, 16, 3)

        self.maxpool1 = nn.MaxPool2d(3)
        self.maxpool2 = nn.MaxPool2d(1)

        x_test = T.zeros(1, input_size[0], input_size[1], input_size[2]).float()
        fc_input_size = self.size_of_conv_out(x_test)

        """ Fully-connected DNN - Dense """

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.f_out = nn.Linear(512, action_size*2) # The actor return a mu and std for every possible action #


    def forward(self, x):
        """ Forward function. """

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = T.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # Action injection
        x = F.relu(self.fc3(x))

        P = self.f_out(x)

        return P

    def size_of_conv_out(self, x):
        """
        Function to extract the output size of the convolutional network.

        :param x: Input of the convolutional network
        :return: Integer with the size of the input of the next layer (FC)
        """

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = T.flatten(x, start_dim=1)

        return x.shape[1]


class Convolutional_CriticNetwork(nn.Module, ABC):
    """
    Convolutional Neural Network for the Critic Q(s,a).
    The Output corresponds with the Q values representing
    the state-action discounted values.

    """

    def __init__(self, input_size, action_size):
        super(Convolutional_CriticNetwork, self).__init__()

        """ First Convolutional part - The state is processed here"""

        self.conv1 = nn.Conv2d(input_size[0], 16, 5)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.maxpool1 = nn.MaxPool2d(3)
        self.maxpool2 = nn.MaxPool2d(1)

        x_test = T.zeros(1, input_size[0], input_size[1], input_size[2]).float()
        fc_input_size = self.size_of_conv_out(x_test)

        """ Fully-connected part - The action is processed here"""

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512 + action_size, 512 + action_size)
        self.fc3 = nn.Linear(512 + action_size, 512 + action_size)
        self.f_out = nn.Linear(512 + action_size, action_size)

    def size_of_conv_out(self, x):

        """
        Function to extract the output size of the convolutional network.

        :param x: Input of the convolutional network
        :return: Integer with the size of the input of the next layer (FC)
        """

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = T.flatten(x, start_dim=1)

        return x.shape[1]

    def forward(self, state, actions):
        """ Forward function. """

        x = F.relu(self.conv1(state))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = T.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(T.cat((x, actions), dim=1))) # Action injection
        x = F.relu(self.fc3(x))
        Q = self.f_out(x)

        return Q