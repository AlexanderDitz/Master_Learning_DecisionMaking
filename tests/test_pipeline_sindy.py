import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy


agent_spice, features, loss = pipeline_sindy.main(
    
    data='data/parameter_recovery/data_32p_0.csv',
    model='params/parameter_recovery/params_32p_0.pkl',
    
    # model = 'params/eckstein2022/params_eckstein2022_test.pkl',
    # data = 'data/eckstein2022/eckstein2022.csv',
    
    # general recovery parameters
    participant_id=2,
    
    # sindy parameters
    polynomial_degree=1,
    optimizer_alpha=0.05,
    optimizer_threshold=0.05,
    n_trials_off_policy=1000,
    n_sessions_off_policy=0,
    verbose=True,
    
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