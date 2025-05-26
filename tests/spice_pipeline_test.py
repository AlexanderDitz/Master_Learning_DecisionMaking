import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn, pipeline_sindy

# path_datasets = 'data/parameter_recovery/data_128p_0.csv'
# path_params = 'params/parameter_recovery_subs/_parameter_recovery_0'

path_datasets = 'data/eckstein2022/eckstein2022.csv'
path_params = 'params/eckstein2022'

if not path_datasets.endswith('.csv'):
    datasets = os.listdir(path_datasets)
else:
    datasets = [path_datasets.split(os.path.sep)[-1]]

losses = []
for d in datasets:
    
    if not path_datasets.endswith('.csv'):
        dataset = os.path.join(path_datasets, d)
    else:
        dataset = path_datasets
    
    name_rnn = d.replace('.csv', f'.pkl')
    if 'data' in d:
        name_rnn = name_rnn.replace('data', 'rnn')
    else:
        name_rnn = 'rnn_' + d
    name_rnn = name_rnn.replace('.csv', '.pkl')
    path_spice_rnn = os.path.join(path_params, name_rnn)
    
    loss_rnn = pipeline_rnn.main(
        epochs=65536,
        scheduler=True,
        learning_rate=1e-4,
        l1_weight_decay=0.001,
        train_test_ratio=1.0,
        
        data=dataset,
        model=path_spice_rnn,
        
        embedding_size=32,
        n_steps=32,
        l2_weight_decay=0,
        dropout=0.25,
        batch_size=-1,
        sequence_length=-1,
        bagging=True,        
        )[-1]
    
    loss_spice = pipeline_sindy.main(
        
        data=dataset,
        model=path_spice_rnn,

        # general SPICE parameters
        participant_id=None,
        filter_bad_participants=False,
        use_optuna=True,
        pruning=True,
        n_trials_off_policy=1000,
        n_trials_same_action_off_policy=5,
        
        # sindy parameters
        optimizer_type="SR3_L1",
        optimizer_alpha=0.1,
        optimizer_threshold=0.05,
        polynomial_degree=1,
        verbose=True,
        
        save = True,
        get_loss = True,
        )[-1]
    
    losses.append((loss_rnn, loss_spice))

print(losses)