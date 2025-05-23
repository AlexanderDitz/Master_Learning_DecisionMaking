import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy


agent_spice, features, loss = pipeline_sindy.main(
    
    # data='data/parameter_recovery/data_32p_0.csv',
    # model='params/parameter_recovery/params_32p_0.pkl',
    
    model = 'params/eckstein2022/rnn_eckstein2022_reward.pkl',
    data = 'data/eckstein2022/eckstein2022.csv',
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
    get_loss=True,
)

print(loss)