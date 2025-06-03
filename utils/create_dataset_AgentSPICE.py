import sys, os

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import create_dataset, get_update_dynamics, BanditsDrift, BanditsFlip_eckstein2022
from utils.setup_agents import setup_agent_spice
from resources.rnn import RLRNN_eckstein2022
from resources.sindy_utils import SindyConfig_eckstein2022


def main(class_rnn, sindy_config, path_rnn, path_spice, path_data, path_save, n_trials_per_session):
    
    # environment = BanditsDrift(sigma=0.2)
    environment = BanditsFlip_eckstein2022()
    
    agent = setup_agent_spice(
        class_rnn=class_rnn,
        path_data=path_data, 
        path_rnn=path_rnn,
        path_spice=path_spice,
        rnn_modules=sindy_config['rnn_modules'],
        control_parameters=sindy_config['control_parameters'],
        sindy_library_setup=sindy_config['library_setup'],
        sindy_filter_setup=sindy_config['filter_setup'],
        sindy_dataprocessing=sindy_config['dataprocessing_setup'],
        sindy_library_polynomial_degree=1,
        deterministic=False,
        verbose=True,
        # participant_id=0,
        )

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
    choice_prob_0, choice_prob_1, action_value_0, action_value_1, reward_value_0, reward_value_1, choice_value_0, choice_value_1 = [], [], [], [], [], [], [], []

    for i in tqdm(range(len(dataset))):    
        # get update dynamics
        experiment = dataset.xs[i].cpu().numpy()
        qs, choice_probs, _ = get_update_dynamics(experiment, agent)
        
        # append behavioral data
        session += list(experiment[:, -1])
        choice += list(np.argmax(experiment[:, :agent._n_actions], axis=-1))
        reward += list(np.max(experiment[:, agent._n_actions:agent._n_actions*2], axis=-1))
        
        # append update dynamics
        choice_prob_0 += list(choice_probs[:, 0])
        choice_prob_1 += list(choice_probs[:, 1])
        action_value_0 += list(qs[0][:, 0])
        action_value_1 += list(qs[0][:, 1])
        reward_value_0 += list(qs[1][:, 0])
        reward_value_1 += list(qs[1][:, 1])
        choice_value_0 += list(qs[2][:, 0])
        choice_value_1 += list(qs[2][:, 1])
        
    columns = ['session', 'choice', 'reward', 'choice_prob_0', 'choice_prob_1', 'action_value_0', 'action_value_1', 'reward_value_0', 'reward_value_1', 'choice_value_0', 'choice_value_1']
    data = np.stack((np.array(session), np.array(choice), np.array(reward), np.array(choice_prob_0), np.array(choice_prob_1), np.array(action_value_0), np.array(action_value_1), np.array(reward_value_0), np.array(reward_value_1), np.array(choice_value_0), np.array(choice_value_1)), axis=-1)
    df = pd.DataFrame(data=data, columns=columns)

    # data_save = path_data.replace('.', '_spice.')
    df.to_csv(path_save, index=False)

    print(f'Data saved to {path_save}')
    

if __name__=='__main__':
    class_rnn = RLRNN_eckstein2022
    sindy_config = SindyConfig_eckstein2022
    path_rnn = 'params/eckstein2022/rnn_eckstein2022_l1_0_001.pkl'
    path_spice = 'params/eckstein2022/spice_eckstein2022_l1_0_001.pkl'
    path_data = 'data/eckstein2022/eckstein2022.csv'
    n_trials_per_session = 500
    
    main(
        class_rnn=class_rnn,
        sindy_config=sindy_config,
        path_rnn=path_rnn,
        path_spice=path_spice,
        path_data=path_data,
        path_save=path_data.replace('.', '_simulated_spice.'),
        n_trials_per_session=n_trials_per_session,
    )