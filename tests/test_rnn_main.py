import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

_, loss = rnn_main.main(
    checkpoint=False,
    epochs=128,
    
    data='data/study_recovery_stepperseverance/data_rldm_256p_0.csv',
    model='params/study_recovery_stepperseverance/params_rldm_256p_0.pkl',
    
    # model=f'params/benchmarking/rnn_eckstein.pkl',
    # data = 'data/2arm/eckstein2022_291_processed.csv',
    
    # model = f'params/benchmarking/rnn_sugawara.pkl',
    # data = 'data/2arm/sugawara2021_143_processed.csv',
    
    n_actions=2,
    
    dropout=0.25,
    participant_emb=True,
    bagging=True,

    learning_rate=5e-3,
    batch_size=-1,
    sequence_length=-1,
    train_test_ratio=1,
    n_steps=16,
    scheduler=True,
    
    n_sessions=256,
    n_trials=256,
    sigma=0.2,
    beta_reward=3.,
    alpha_reward=0.25,
    alpha_penalty=0.25,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=0.,
    alpha_choice=0.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    analysis=True,
    participant_id=5,
)