import sys, os

import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import create_dataset, get_update_dynamics, AgentQ, BanditsDrift, BanditsFlip_eckstein2022
from resources.rnn_utils import DatasetRNN
from utils.convert_dataset import convert_dataset

# dataset specific benchmarking models
from benchmarking import benchmarking_dezfouli2019, benchmarking_eckstein2022, benchmarking_lstm

# ------------------- CONFIGURATION ECKSTEIN2022 w/o AGE --------------------
dataset = 'eckstein2022'
model = 'ApAnBrBcfBch'
# model = 'ApAnBrBcfBch'
setup_agent = benchmarking_eckstein2022.setup_agent_benchmark
rl_model = benchmarking_eckstein2022.rl_model
bandits_environment = BanditsFlip_eckstein2022
path_model = f'params/{dataset}/mcmc_{dataset}_{model}.nc'

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
# dataset = 'dezfouli2019'
# # model = 'PhiBeta'
# model = 'PhiChiBetaKappaC'
# setup_agent = benchmarking_dezfouli2019.setup_agent_gql
# Dezfouli2019GQL = benchmarking_dezfouli2019.Dezfouli2019GQL
# bandits_environment = BanditsDrift
# path_model = f'params/{dataset}/gql_{dataset}_{model}.pkl'

n_trials_per_session = 200

path_data = f'data/{dataset}/{dataset}.csv'
path_save = f'data/{dataset}/{dataset}_simulated_{model}.csv'


# --------------------- PIPELINE -------------------------------------

dataset = convert_dataset(file=path_data)[0]
participant_ids = dataset.xs[:, 0, -1].unique().int().numpy()

xs = torch.zeros((len(participant_ids), n_trials_per_session, dataset.xs.shape[-1]))
ys = torch.zeros((len(participant_ids), n_trials_per_session, dataset.ys.shape[-1]))

agent = setup_agent(path_model, deterministic=False, model_config=model)

for participant_id in tqdm(participant_ids):
    environment = bandits_environment(sigma=0.2)
    dataset = create_dataset(
                agent=agent[0][participant_id],
                environment=environment,
                n_trials=n_trials_per_session,
                n_sessions=1,
                verbose=False,
                )[0]
    
    dataset.xs[..., -1] += participant_id
    
    xs[participant_id, :, :dataset.xs.shape[-1]-1] = dataset.xs[0, :, :-1]
    xs[participant_id, :, -1] = dataset.xs[0, :, -1]
    ys[participant_id] = dataset.ys[0]

dataset = DatasetRNN(xs, ys)

# dataset columns
# general dataset columns
session, choice, reward = [], [], []

print('Saving values...')
for index_data in tqdm(range(len(dataset))):
    session += list(dataset.xs[index_data, :, -1].detach().cpu().numpy())
    choice += list(np.argmax(dataset.xs[index_data, :, :agent[0][0]._n_actions].detach().cpu().numpy(), axis=-1))
    reward += list(np.max(dataset.xs[index_data, :, agent[0][0]._n_actions:agent[0][0]._n_actions*2].detach().cpu().numpy(), axis=-1))
    
columns = ['session', 'choice', 'reward']
data = np.stack((np.array(session), np.array(choice), np.array(reward)), axis=-1)
df = pd.DataFrame(data=data, columns=columns)

# data_save = path_data.replace('.', '_'+model+'.')
df.to_csv(path_save, index=False)

print(f'Data saved to {path_save}')