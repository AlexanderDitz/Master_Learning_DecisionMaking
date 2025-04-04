import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn


_, loss = pipeline_rnn.main(
    checkpoint=False,
    # epochs=32768,
    epochs=1,
    
    data='data/parameter_recovery/data_128p_0.csv',
    model='params/parameter_recovery/params_128p_0_toyrun.pkl',
    
    # data = 'data/optuna/data_128p_0.csv',
    # model = 'params/params_128p_0.pkl',
    
    # model=f'params/eckstein2022/params_eckstein2022.pkl',
    # data=f'data/eckstein2022/eckstein2022.csv',
    
    # model=f'params/sugawara2021/params_sugawara2021.pkl',
    # data=f'data/sugawara2021/sugawara2021.csv',
    
    n_actions=2,
    
    # final params
    # embedding_size=22,
    # n_steps=32,
    # learning_rate=0.00023,
    # train_test_ratio=0.8,
    # scheduler=True,
    
    # toy params for quick run
    learning_rate=5e-3,
    n_steps=16,
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
    beta_reward=3.,
    alpha_reward=0.25,
    alpha_penalty=0.25,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    analysis=True,
    participant_id=0,
)