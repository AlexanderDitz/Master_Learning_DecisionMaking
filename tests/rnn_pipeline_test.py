import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_eckstein2022, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_rearranged


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------

# path_model = 'params/eckstein2022/rnn_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'
# path_data = 'data/eckstein2022/eckstein2022_age.csv'
# train_test_ratio = 0.8
# class_rnn = RLRNN_eckstein2022
# additional_inputs = None
# # class_rnn = RLRNN_meta_eckstein2022
# # additional_inputs = ['age']

class_rnn = RLRNN_dezfouli2019
train_test_ratio = [3, 6, 9]  # list of test sessions
# path_model = 'params/dezfouli2019/rnn_dezfouli2019_rldm_l1emb_0_001_l2_0_0005.pkl'
path_model = 'params/dezfouli2019/rnn_dezfouli2019_test.pkl'
path_data = 'data/dezfouli2019/dezfouli2019.csv'
additional_inputs = None

# -------------------------------------------------------------------------------
# SPICE PIPELINE
# -------------------------------------------------------------------------------

_, loss = pipeline_rnn.main(
    
    # sparsification parameter
    l1_weight_decay=0.001,
    # generalization parameters
    l2_weight_decay=0.0005,
    dropout=0.25,
    train_test_ratio=train_test_ratio,
    
    # general training parameters
    checkpoint=False,
    epochs=1024, # <- 2^16
    scheduler=False,
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
    beta_reward=1.,
    alpha_reward=0.25,
    alpha_penalty=0.25,
    forget_rate=0.3,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    save_checkpoints=True,
    analysis=True,
    participant_id=0,
)