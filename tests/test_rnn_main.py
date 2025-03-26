import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

_, loss = rnn_main.main(
    checkpoint=False,
    epochs=4096,
    
    data='data/parameter_recovery_participants/data_128p_0.csv',
    model='params/parameter_recovery_participants/params_128p_0.pkl',
    
    # model=f'params/benchmarking/rnn_eckstein.pkl',
    # data = 'data/2arm/eckstein2022_291_processed.csv',
    
    # model = f'params/benchmarking/rnn_sugawara.pkl',
    # data = 'data/2arm/sugawara2021_143_processed.csv',
    
    n_actions=2,
    
    dropout=0.31905877051758935,#0.25,
    hidden_size=22,#8,
    embedding_size=9,#8,

    learning_rate=5e-3,#0.0003638666906537081,#
    batch_size=-1,
    sequence_length=-1,
    train_test_ratio=1,
    n_steps=62,#16,
    scheduler=True,
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
    participant_id=1,
)