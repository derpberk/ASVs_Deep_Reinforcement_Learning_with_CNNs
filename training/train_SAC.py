
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from agents.Soft_AC import Soft_Actors_Critic
import numpy as np
from environments.IPP_Ypacarai import DiscreteIPP
from utils.config import Configuration
from torch.cuda import is_available as check_if_cuda_available
from torch.cuda import get_device_name
from utils.logging_utils import *
import signal

keep_going = True

console = create_console()
console.clear()

""" Create configure structure """
config = Configuration()

""" Configure seed """
config.seed = 0


""" Create scenario """
my_map = np.genfromtxt(sys.path[0] + '/example_map.csv', delimiter=',')
config.environment = DiscreteIPP(domain_type='Continuous',
                                 scenario_map=my_map,
                                 number_of_features=100,
                                 detection_area_ratio=4)

config.state_size = config.environment.reset().shape
config.action_size = config.environment.action_size

""" Configure device """
if check_if_cuda_available():
    config.device = 'cuda:0'
    config.device_name = get_device_name(0)
else:
    config.device = 'cpu'

""" Configure simulation conditions """
config.number_of_episodes = 10

""" Hyperparameters """
config.hyperparameters = {
    'buffer_size': 10000,
    'batch_size': 64,
    'seed': config.seed,
    'mu': np.array([0]),
    'add_extra_noise': False,
    'sigma': 0,
    'theta': 0,
    'dt': 0,
    'training_episodes_per_eval_episodes': 10,
    'initial_random_steps_number': 50,
    'initial_epsilon': 1,
    'EPSILON': 1E-6,
    'tau': 0.005,
    'discount_rate': 0.95,
    'gradient_clipping_norm': None,
    'Actor': {
        'learning_rate': 0.0001
    },
    'Critic': {
        'learning_rate': 0.0001
    }
}
present_hyperparameters(console, hyperparameters=config.hyperparameters)

""" Create Agent """
Agent = Soft_Actors_Critic(config)

Agent.reset()

def my_shutdown_handler(signum, frame):
    # set a flag to exit your loop or call what you need to exit gracefully
    global keep_going
    keep_going = False
    print('ENDING TRAINING!')


""" Main loop """
def main():

    for i in range(config.number_of_episodes):

        if not keep_going:
            break

        progress = Agent.training_step()
        print_progress(console, progress)

        if i % int(config.number_of_episodes/10) and i != 0:
            Agent.save_progress(save_models=False)

    Agent.save_progress(save_models=True)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, my_shutdown_handler)
    main()















