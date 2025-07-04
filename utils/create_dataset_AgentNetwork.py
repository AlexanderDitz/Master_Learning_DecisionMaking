import sys, os

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import create_dataset, get_update_dynamics, BanditsDrift, BanditsFlip_eckstein2022
from utils.setup_agents import setup_agent_rnn

# dataset specific SPICE configurations and models
from resources.rnn import RLRNN, RLRNN_eckstein2022, RLRNN_dezfouli2019
from resources.sindy_utils import SindyConfig, SindyConfig_eckstein2022, SindyConfig_dezfouli2019

# ------------------- CONFIGURATION ECKSTEIN2022 w/o AGE --------------------
# dataset = 'eckstein2022'
# class_rnn = RLRNN_eckstein2022
# sindy_config = SindyConfig_eckstein2022
# bandits_environment = BanditsFlip_eckstein2022

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
dataset = 'dezfouli2019'
class_rnn = RLRNN_eckstein2022
sindy_config = SindyConfig_eckstein2022
bandits_environment = BanditsDrift


# ----------------------- GENERAL CONFIGURATION ----------------------------
n_trials_per_session = 200

path_data = f'data/{dataset}/{dataset}.csv'
path_rnn = f'params/{dataset}/rnn_{dataset}_l2_0_0005.pkl'
path_save = f'data/{dataset}/{dataset}_simulated_rnn_test.csv'


# ------------------- PIPELINE ----------------------------
environment = bandits_environment(sigma=0.2)

agent = setup_agent_rnn(
    class_rnn=class_rnn,
    path_model=path_rnn,
    deterministic=False,
    )

print('Creating dataset...')
dataset, _, _ = create_dataset(
            agent=agent,
            environment=environment,
            n_trials=n_trials_per_session,
            n_sessions=agent._model.n_participants,
            verbose=False,
            )

# dataset columns
# general dataset columns
session, choice, reward = [], [], []

print('Saving values...')
for i in tqdm(range(len(dataset))):    
    # get update dynamics
    experiment = dataset.xs[i].cpu().numpy()
    qs, choice_probs, _ = get_update_dynamics(experiment, agent)
    
    # append behavioral data
    session += list(experiment[:, -1])
    choice += list(np.argmax(experiment[:, :agent._n_actions], axis=-1))
    reward += list(np.max(experiment[:, agent._n_actions:agent._n_actions*2], axis=-1))
    
columns = ['session', 'choice', 'reward']
data = np.stack((np.array(session), np.array(choice), np.array(reward)), axis=-1)
df = pd.DataFrame(data=data, columns=columns)

df.to_csv(path_save, index=False)

print(f'Data saved to {path_save}')