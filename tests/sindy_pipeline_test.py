import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy
from resources.rnn import RLRNN, RLRNN_eckstein2022, RLRNN_dezfouli2019, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_trials
from resources.sindy_utils import SindyConfig, SindyConfig_eckstein2022, SindyConfig_dezfouli2019, SindyConfig_eckstein2022_trials


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------

# path_data='data/parameter_recovery/data_256p_0.csv'
# path_model='params/parameter_recovery/rnn_256p_0.pkl'
# class_rnn = RLRNN
# sindy_config = SindyConfig

path_data = 'data/eckstein2022/eckstein2022_age.csv'
path_model = 'params/eckstein2022/rnn_eckstein2022_age.pkl'
# class_rnn = RLRNN_eckstein2022
class_rnn = RLRNN_meta_eckstein2022
sindy_config = SindyConfig_eckstein2022
additional_inputs = ['age']

# path_data = 'data/dezfouli2019/dezfouli2019.csv'
# path_model = 'params/dezfouli2019/rnn_dezfouli2019.pkl'
# class_rnn = RLRNN_dezfouli2019
# sindy_config = SindyConfig_dezfouli2019


# -------------------------------------------------------------------------------
# SPICE PIPELINE
# -------------------------------------------------------------------------------

agent_spice, features, loss = pipeline_sindy.main(
    
    class_rnn=class_rnn,
    model = path_model,
    data = path_data,
    additional_inputs_data=additional_inputs,
    save = True,
    
    # general recovery parameters
    participant_id=None,
    filter_bad_participants=False,
    use_optuna=True,
    pruning=True,
    
    # sindy parameters
    # optimizer_type="SR3_weighted_l1",
    train_test_ratio=0.8,
    polynomial_degree=1,
    optimizer_alpha=0.1,
    optimizer_threshold=0.05,
    n_trials_off_policy=1000,
    n_sessions_off_policy=1,
    n_trials_same_action_off_policy=5,
    optuna_threshold=0.05,
    optuna_trials_first_state=20,
    optuna_trials_second_state=20,
    verbose=False,
    
    # generated training dataset parameters
    n_actions=2,
    sigma=0.2,
    beta_reward=1.,
    alpha=0.25,
    alpha_penalty=0.25,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    analysis=True,
    get_loss=False,
    
    **sindy_config,
)

print(loss)