from abc import ABC
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Convolutional_Q_Network(nn.Module, ABC):

    def __init__(self, input_size, action_size):
        super(Convolutional_Q_Network, self).__init__()

        """ Convolutional DNN """

        self.conv1 = nn.Conv2d(input_size[0], 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 3)

        x_test = T.zeros(1, input_size[0], input_size[1], input_size[2]).float()
        fc_input_size = self.size_of_conv_out(x_test)

        """ Fully-connected DNN - Dense """

        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.f_out = nn.Linear(512, action_size)

    def forward(self, x):
        """ Forward function. """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = T.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        Q = self.f_out(x)

        return Q

    def size_of_conv_out(self, x):
        """
        Function to extract the output size of the convolutional network.

        :param x: Input of the convolutional network
        :return: Integer with the size of the input of the next layer (FC)
        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = T.flatten(x, start_dim=1)

        return x.shape[1]
