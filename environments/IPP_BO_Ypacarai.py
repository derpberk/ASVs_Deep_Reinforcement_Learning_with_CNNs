import gym
import matplotlib.pyplot as plt
import numpy as np
from environments.groundtruthgenerator import GroundTruth
import warnings
warnings.simplefilter("ignore", UserWarning)
from skopt.learning.gaussian_process import gpr, kernels


class ContinuousBO(gym.Env):
    environment_name = "Continuous Informative Path Planning"

    def __init__(self, scenario_map, initial_position=None, battery_budget=100, resolution=1, seed=0):

        self.id = "Continuous BO Ypacarai"

        # Map of the environment #
        self.scenario_map = scenario_map

        # Environment boundaries
        self.map_size = self.scenario_map.shape
        self.map_lims = np.array(self.map_size) - 1

        # Action space and sizes #
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]))
        self.action_size = 2

        # Internal state of the Markov Decision Process #
        self.state = None
        self.next_state = None
        self.reward = None
        self.done = False

        # Create the ground truth object #
        self.gt = GroundTruth(1 - self.scenario_map, 1, max_number_of_peaks=5)

        # Initial position, referred as a [X,Y] vector #
        self.initial_position = initial_position
        self.step_count = 0

        # Penalization for a collision in the reward funcion #
        self.collision_penalization = 10

        # Seed for deterministic replication #
        self.seed = seed
        np.random.seed(self.seed)

        # Battery budget
        self.battery = battery_budget

        # Battery cost -> Cost of the battery per 1m movement
        # This is calculated approximately using the long size of the map ~ 15km in the Ypacarai case
        self.battery_cost = 100 / np.max(self.map_size) / resolution

        # Gaussian Process parameters
        # GP with RBF kernel of 10% long-size of the map lengthscale (in pixels)
        self.gp = gpr.GaussianProcessRegressor(kernel=kernels.RBF(0.1 * np.min(self.map_size)), alpha=1e-7)

        # Matrix de (num_muestra, features): num_muestra: numero de posiciones en la que se tomaron muestras
        #                                    features: cada una de las dimenciones, i.e., features son y, x
        self.train_inputs = None

        # Solution vector: y = f(x)
        self.train_targets = None

        #  Matrix de (num_pos, features): num_pos: numero de posiciones en la que puede encontrarse el ASV
        #                                 features: cada una de las dimenciones, i.e., features son y, x
        # [[2 17]
        #  [2 18]
        # ...
        #  [y x]
        # ...
        # [54 31]
        # [54 32]]
        self.possible_locations = np.asarray(np.where(self.scenario_map == 1)).reshape(2, -1).T

        # Vector!! de dimension 1 fila x (mxn) columnas representando el mapa de incertidumbres anterior
        self.old_std = None
        self._max_step_distance = np.min(self.map_size)

        # Initial position #
        # The initial position HAS PIXEL UNITS:
        #    The action spaces translates to PIXEL UNITS TO FORM THE STATE
        self.position = None
        self.place_agent()
        self.reset()

    def max_step_distance(self):
        # esto se puede convertir en funcion del length scale
        return 0.2 * self._max_step_distance
        # return  lambda * np.exp(self.gp.kernel_.theta[0])

    def reset(self):
        """ Reset the internal parameters of the environment. """

        # Place the agent in the initial point depending if it is defined a fixed initial position.
        self.place_agent()

        # Reset the battery
        self.battery = 100

        # Reset the internal MPD variables
        self.reward = None
        self.done = False
        self.next_state = None
        self.step_count = 0

        # Generate another gt -> changes internally self.gt.normalized_z
        # normalized_z is also achievable using self.gt.read()
        self.gt.reset()

        # Reset Gaussian Process parameters #
        # Generate the first input X
        self.train_inputs = np.array([self.position]).reshape(-1, 2)
        # Evaluate the environment in this particular initial point
        self.train_targets = np.array([self.gt.normalized_z[self.position[0], self.position[1]]])
        # Fit the Gaussian Process
        self.gp.fit(self.train_inputs, self.train_targets)
        # Generate the uncertainty map
        _, self.old_std = self.gp.predict(self.possible_locations, return_std=True)
        # Process the state
        self.process_state()

        return self.state

    def place_agent(self):
        """ Place the agent in a random place. """
        if self.initial_position is None:
            indx = np.random.randint(0, len(self.possible_locations))
            self.position = self.possible_locations[indx]
        else:
            self.position = np.copy(self.initial_position)

    def process_state(self):
        """ Process the state """

        """
        state[0] -> position
        state[1] -> boundaries
        state[2] -> features
        """

        state = np.zeros(shape=(3, self.scenario_map.shape[0], self.scenario_map.shape[1])).astype(float)

        # State - position #
        state[0, self.position[0], self.position[1]] = 1

        # State - boundaries #
        state[1] = np.copy(self.scenario_map)

        # State - old standard deviation
        state[2][self.possible_locations[:, 0], self.possible_locations[:, 1]] = self.old_std

        self.state = state

    def render(self, **kwargs):
        """ Render the state for visualization purposes. Outputs the stacked rgb resultant. """

        red = np.copy(self.state[1]) + (1 - self.scenario_map)
        green = np.copy(self.state[2]) + (1 - self.scenario_map)
        blue = np.copy(self.state[0]) + (1 - self.scenario_map)

        rgb = np.stack((red, green, blue), axis=-1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3))

        axs[0].imshow(self.state[0])
        axs[0].set_title('Position')
        axs[1].imshow(self.state[1])
        axs[1].set_title('Navigation map')
        axs[2].imshow(self.state[2])
        axs[2].set_title('$\\sigma(x)$')

        plt.show()

        return rgb

    def action2vector(self, desired_action):
        """ Translate a desired action into a pixel velocity vector. """
        desired_distance = self.max_step_distance() * desired_action[0]
        desired_angle = 2 * 3.141592 * desired_action[1]
        return np.array(
            [-desired_distance * np.sin(desired_angle),
             desired_distance * np.cos(desired_angle)]
        )

    def step(self, desired_action):
        """ Process an action, generates the new state and the reward to that action. """

        self.step_count += 1
        next_position = self.action2vector(desired_action) + self.position  # The next intended position
        next_position = np.clip(next_position, (0, 0), self.map_lims)  # Clip the intended position to be inside the map
        next_position = np.floor(next_position).astype(int)  # Discrete

        if self.scenario_map[next_position[0], next_position[1]] == 1: # If the next position is navigable ...
            valid = True
        else:
            valid = False

        if valid:
            distance = np.linalg.norm(next_position - self.position)  # Compute the intended travel distance IN PIXELS
            self.position = next_position  # Update the position
            self.battery -= distance * self.battery_cost  # Compute the new battery level
            self.train_inputs = np.vstack([self.train_inputs, self.position])  # Store the new sampling point
            self.train_targets = np.append(self.train_targets, self.gt.normalized_z[self.position[0], self.position[1]])
            self.gp.fit(self.train_inputs, self.train_targets)  # Fit the stored sampled points

        else:
            distance = np.linalg.norm(next_position - self.position)  # If not valid, it consumes the intended battery
            self.battery -= distance * self.battery_cost

        self.compute_reward(valid)  # Reward function evaluation
        self.process_state()  # Generate the new state

        # Check the episodic-end condition
        self.done = self.battery <= self.battery_cost

        return self.state, self.reward, self.done, None

    def compute_reward(self, valid):

        r = 0

        if not valid:
            r -= self.collision_penalization

        else:
            _, std = self.gp.predict(self.possible_locations, return_std=True)
            r = np.sum(std - self.old_std)/np.sum(self.old_std)
            self.old_std = std
        self.reward = r


if __name__ == "__main__":

    """ Test to check the wall-time for an episode to run and the average number of steps per episode """

    my_map = np.genfromtxt('YpacaraiMap_big.csv', delimiter=',').astype(int)/255
    env = ContinuousBO(scenario_map=my_map, resolution = 1)
    env.render()

    import time
    t0 = time.time()

    for i in range(100):
        env.reset()
        d = False
        print('Episode ',i)

        while not d:

            a = np.random.rand(2)
            s, r_, d, _ = env.step(a)

        print('Number of steps: ', env.step_count)


    print((time.time() - t0)/100,' segundos la iteracion')
