import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_eckstein2022, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_rearranged


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------

# class_rnn = RLRNN_meta_eckstein2022
class_rnn = RLRNN_eckstein2022
path_model = 'params/eckstein2022/rnn_eckstein2022_lr_0_01_d_0_l1_0.pkl'
path_data = 'data/eckstein2022/eckstein2022_age.csv'
additional_inputs = None#['age']

# class_rnn = RLRNN_dezfouli2019
# path_model = 'params/dezfouli2019/rnn_dezfouli2019_test.pkl'
# path_data = 'data/dezfouli2019/dezfouli2019.csv'
# additional_inputs = None

# -------------------------------------------------------------------------------
# SPICE PIPELINE
# -------------------------------------------------------------------------------

_, loss = pipeline_rnn.main(
    
    checkpoint=False,
    epochs=65536, # <- 2^16
    scheduler=True,
    learning_rate=1e-2,
    l1_weight_decay=0,
    train_test_ratio=1.0,
    n_steps=-1,
    dropout=0,
    save_checkpoints=True,
    
    class_rnn=class_rnn,
    model=path_model,
    data=path_data,
    additional_inputs_data=additional_inputs,
    
    # hand-picked params
    embedding_size=32,
    l2_weight_decay=0,
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
    participant_id=0,
)