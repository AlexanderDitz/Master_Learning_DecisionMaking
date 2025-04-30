import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn


_, loss = pipeline_rnn.main(
    checkpoint=True,
    epochs=0,#65536,
    
    # data='data/parameter_recovery/data_128p_0.csv',
    # model='params/parameter_recovery_2/params_128p_0.pkl',
    
    # data = 'data/optuna/data_128p_0.csv',
    # model = 'params/params_128p_0.pkl',
    
    # model=f'params/eckstein2022/params_eckstein2022.pkl',
    # data=f'data/eckstein2022/eckstein2022.csv',
    
    n_actions=2,
    
    # final params
    # embedding_size=22,
    # n_steps=32,
    # learning_rate=0.00023,
    # train_test_ratio=1.0,
    # scheduler=True,
    
    # toy params for quick run
    learning_rate=1e-3,
    n_steps=32,
    embedding_size=0,
    train_test_ratio=1.0,
    scheduler=False,
    
    # other training parameters
    batch_size=-1,
    sequence_length=-1,
    bagging=True,
    
    n_sessions=128,
    n_trials=200,
    sigma=0.2,
    beta_reward=1.,
    alpha_reward=0.25,
    alpha_penalty=0.25,
    forget_rate=0.3,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    analysis=True,
    participant_id=2,
)