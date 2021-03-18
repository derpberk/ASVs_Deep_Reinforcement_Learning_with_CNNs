import gym
import numpy as np
import matplotlib.pyplot as plt
from deap.benchmarks import shekel
import time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

class DiscreteIPP(gym.Env):

    environment_name = "Discrete/Continuous Informative Path Planning"
    
    def __init__(self, domain_type, scenario_map, number_of_features=10, initial_position=None, battery_budget=100,
                 obstacles=False, resolution = 1, detection_area_ratio = 2, distribution = 'uniform', random_features=True, seed = 0, max_step_distance = 4):

        self.id = "Discrete Ypacarai"
        # Map of the environment #
        self.scenario_map = scenario_map
        self.map_size = self.scenario_map.shape
        self.map_lims = np.array(self.map_size) - 1
        self.domain_type = domain_type

        if domain_type == 'Discrete':
            self.action_space = gym.spaces.Discrete(8)
            self.action_size = 8
        elif domain_type == 'Full Continuous':
            self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]))
            self.action_size = 2
        else:
            self.action_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]))
            self.action_size = 1

        self.detected_area = detection_area_ratio
        self.state = None
        self.next_state = None
        self.reward = None
        self.done = False
        self.distribution = distribution
        self.initial_position = initial_position
        self.step_count = 0
        self.collision_penalization = 10
        self.seed = seed
        np.random.seed(self.seed)
        self.episodic_reward = []
        self.reward_threshold = float("inf")
        self.trials = 50
        self.visual = True
        self.random_features = random_features
        self.max_step_distance = max_step_distance

        # Initialization of the random map information features #

        self.number_of_features = number_of_features
        self.place_features()
        self.idleness = np.ones(self.number_of_features)
        # TODO: Hay que definir el criterio de importancia finalmente
        self.features_importance = np.linspace(0, 1, self.number_of_features)

        # Initial position #
        # The initial position HAS PIXEL UNITS:
        #    The action spaces translates to PIXEL UNITS TO FORM THE STATE
        self.position = None
        self.place_agent()

        # Battery budget
        self.battery = battery_budget

        # Battery cost -> Cost of the battery per 1m movement
        # TODO: De momento, el coste de movimiento unitario está en el cociente entre el total de
        #  batería disponible y la  distancia total que se puede recorrer: dar una vuelta al perimetro

        self.battery_cost = 100/(2*self.map_size[0] + 2*self.map_size[1])/resolution
        self.recovery_rate = self.battery_cost/100/2

        self.reset()

    def reset(self):

        self.place_agent()
        if self.random_features is True:
            self.place_features()
        self.battery = 100
        self.reward = None
        self.done = False
        self.process_state()
        self.next_state = None
        self.step_count = 0
        self.episodic_reward = []
        self.idleness = np.ones(self.number_of_features)

        return self.state

    def place_agent(self):
        """ Place the agent in a random place. """
        if self.initial_position is None:
            posibles = np.asarray(np.nonzero(self.scenario_map)).T
            indx = np.random.randint(0, len(posibles))
            self.position = posibles[indx]
        else:
            self.position = np.copy(self.initial_position)

    def place_features(self):
        """ Place the features """

        if self.distribution == 'uniform':
            ij = np.asarray(np.nonzero(self.scenario_map)).T
            indx = np.random.randint(0, len(ij), size=self.number_of_features)
            self.features_positions = ij[indx]

    def process_state(self):
        """ Process the state """

        """
        state[0] -> position
        state[1] -> boundaries
        state[2] -> features
        state[3] -> coverage_area
        """

        state = np.zeros(shape=(4,self.scenario_map.shape[0],self.scenario_map.shape[1])).astype(float)

        # State - position #
        state[0,self.position[0],self.position[1]] = 1

        # State - boundaries #
        state[1] = np.copy(self.scenario_map)

        # State - Features #
        for n, pos in enumerate(self.features_positions):
            state[2, pos[0], pos[1]] = self.idleness[n]*self.features_importance[n]

        # State - converage area #
        state[3,
            np.clip(self.position[0]-self.detected_area, 0, self.map_size[0]):
            np.clip(self.position[0]+self.detected_area + 1, 0, self.map_size[0]),
            np.clip(self.position[1]-self.detected_area, 0, self.map_size[1]):
            np.clip(self.position[1]+self.detected_area + 1, 0, self.map_size[1])]\
            = 1

        self.state = state

    def render(self):

        red = np.copy(self.state[2]) + (1-self.scenario_map)
        green = np.copy(self.state[3]) + (1-self.scenario_map)
        blue = np.copy(self.state[0]) + (1-self.scenario_map)

        rgb = np.stack((red, green, blue), axis=-1)
        fig, axs = plt.subplots(1, 5, figsize=(15 , 3))

        axs[0].imshow(self.state[0])
        axs[0].set_title('Position')
        axs[1].imshow(self.state[1])
        axs[1].set_title('Navigation map')
        axs[2].imshow(self.state[2])
        axs[2].set_title('Features')
        axs[3].imshow(self.state[3])
        axs[3].set_title('Coverage area')
        axs[4].imshow(rgb)
        axs[4].set_title('RGB image')

        plt.show()

        return rgb

    def step(self, desired_action):

        self.step_count += 1

        if self.domain_type == 'Discrete':
            # Check the feasibility of the action #
            valid, new_position = self.check_action(desired_action)

            if valid:  # Valid action
                # IF valid, update the position
                self.position = new_position

            self.process_state()  # State is processed #
            self.reward = self.compute_reward(valid)
            self.episodic_reward.append(self.reward)

            # Compute the battery consumption #
            self.battery -= self.battery_cost if desired_action < 4 else 1.4142 * self.battery_cost

            # Check the episodic end condition
            self.done = self.battery <= self.battery_cost

            # Recover the idleness of the features #
            self.idleness = np.clip(self.idleness + self.recovery_rate, 0, 1)

        else:
            """ For the continuous action there are two control signals: 
                [0] -> distance
                [1] -> angle """

            if self.domain_type == 'Full Continuous':
                desired_distance = self.max_step_distance * desired_action[0]
                desired_angle = 2*3.141592*desired_action[1]
            else:
                desired_distance = self.max_step_distance
                desired_angle = 2 * 3.141592 * desired_action

            next_position_local = np.array(
                [desired_distance * np.sin(desired_angle),
                 desired_distance * np.cos(desired_angle)]
            )

            next_position = np.array([-1, 1])*next_position_local + self.position

            next_position = np.clip(next_position, (0,0), self.map_lims)

            next_position = np.floor(next_position).astype(int)

            if self.scenario_map[next_position[0], next_position[1]] == 1:
                valid = True
            else:
                valid = False

            if valid:
                distance = np.linalg.norm(next_position-self.position)
                self.position = next_position
                self.battery -= distance * self.battery_cost
            else:
                distance = np.linalg.norm(next_position - self.position)
                self.battery -= distance * self.battery_cost

            self.process_state()
            self.reward = self.compute_reward(valid)
            self.episodic_reward.append(self.reward)

            # Check the episodic end condition
            self.done = self.battery <= self.battery_cost

            # Recover the idleness of the features #
            self.idleness = np.clip(self.idleness + self.recovery_rate * distance, 0, 1)

        return self.state, self.reward, self.done, None

    def compute_reward(self, valid):

        r = 0

        if not valid:
            r -= self.collision_penalization

        else:

            # Compute the gain of the surroundings #
            for indx, pos in enumerate(self.features_positions):

                if (self.position[0]-self.detected_area <= pos[0] <= self.position[0]+self.detected_area) \
                        and (self.position[1]-self.detected_area <= pos[1] <= self.position[1]+self.detected_area):

                    r += self.idleness[indx]*self.features_importance[indx]

                    # Diminish the next reward #
                    self.idleness[indx] = 0

        self.reward = r

        return r

    def check_action(self, desired_action):

        v = self.action2vector(desired_action)
        valid = False

        attempted_position = self.position + v

        if self.scenario_map[attempted_position[0],attempted_position[1]] == 1:
            valid = True
        else:
            valid = False

        return valid, attempted_position

    @staticmethod
    def action2vector(action):

        if action == 0:
            vector = np.asarray([-1, 0])
        elif action == 1:
            vector = np.asarray([0, 1])
        elif action == 2:
            vector = np.asarray([1, 0])
        elif action == 3:
            vector = np.asarray([0, -1])
        elif action == 4:
            vector = np.asarray([-1, 1])
        elif action == 5:
            vector = np.asarray([1, 1])
        elif action == 6:
            vector = np.asarray([1, -1])
        elif action == 7:
            vector = np.asarray([-1, -1])

        return vector


if __name__ == "__main__":

    my_map = np.genfromtxt('example_map.csv', delimiter=',')
    env = DiscreteIPP(domain_type='Discrete', scenario_map=my_map, number_of_features = 150, detection_area_ratio=4)
    env.render()

    while not env.done:

        a = int(input())
        s,r,d,_ = env.step(a)
        print(r,d)
        env.render()







    








