import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import Agent, AgentQ, BanditsDrift, BanditsSwitch, plot_session, create_dataset, get_update_dynamics
from utils.plotting import plot_session
from benchmarking import benchmarking_dezfouli2019
from utils.convert_dataset import convert_dataset


path_data = 'data/dezfouli2019/dezfouli2019_simulated_baseline.csv'

agent1 = Agent(
    beta_reward=1.,
    alpha_reward=0.5,
    )

agent2 = benchmarking_dezfouli2019.Agent_dezfouli2019(
    n_actions=2,
    d=2,
    phi=np.array([0.145, 0.815]),
    chi=np.array([0.635, 0.389]),
    beta=np.array([4.258, -1.002]),
    kappa=np.array([3.268, -0.974]),
    C=np.array([[-14.256, 4.243],[17.998, -6.335]]),
    deterministic=False,
)

# env = EnvironmentBanditsSwitch(0.05, reward_prob_high=1.0, reward_prob_low=0.5)
# env = BanditsDrift(0.2)
# trajectory = create_dataset(agent1, env, 128, 1)[0]
trajectory = convert_dataset(path_data)[0]

agents = {
    'groundtruth': agent1, 
    'benchmark': agent2,
    }
fig, axs = plot_session(agents, trajectory.xs[0])
plt.show()