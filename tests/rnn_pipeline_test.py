import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn
from resources import rnn

# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------

class_rnn = rnn.RLRNN_eckstein2022
additional_inputs = None

path_model = 'params/eckstein2022/rnn_eckstein2022_no_l1_l2_0.pkl'
path_data = 'data/eckstein2022/eckstein2022.csv'
train_test_ratio = 0.8

# path_model = 'params/dezfouli2019/rnn_dezfouli2019_no_l1_l2_0_001.pkl'
# path_data = 'data/dezfouli2019/dezfouli2019.csv'
# train_test_ratio = [3, 6, 9]#[1, 3, 4, 6, 8, 10]   # list of test sessions

# train_test_ratio = 1.0
# path_model = 'params/rnn_test_ep256.pkl'
# path_data = None

# -------------------------------------------------------------------------------
# SPICE PIPELINE
# -------------------------------------------------------------------------------

_, loss = pipeline_rnn.main(
    
    # sparsification parameter
    l1_weight_decay=0.,
    # generalization parameters
    l2_weight_decay=0.,
    dropout=0.5,
    train_test_ratio=train_test_ratio,
    
    # general training parameters
    checkpoint=False,
    epochs=65536, # <- 2^16
    scheduler=True,
    learning_rate=1e-2,
    
    # hand-picked params
    n_steps=-1,
    embedding_size=32,
    batch_size=-1,
    sequence_length=-1,
    bagging=True,
    
    class_rnn=class_rnn,
    model=path_model,
    data=path_data,
    additional_inputs_data=additional_inputs,
    
    # synthetic dataset parameters
    n_sessions=128,
    n_trials=200,
    sigma=0.2,
    beta_reward=3.,
    alpha_reward=0.25,
    alpha_penalty=0.5,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=0.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    save_checkpoints=True,
    analysis=False,
    participant_id=5,
)