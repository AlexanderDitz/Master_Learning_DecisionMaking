import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

path_datasets = 'data/parameter_recovery/'
path_params = 'params/parameter_recovery/'

datasets = os.listdir(path_datasets)

losses = []
for d in datasets:
    dataset = os.path.join(path_datasets, d)
    model = os.path.join(path_params, d.replace('.csv', f'.pkl').replace('data', 'params'))
    
    _, loss = rnn_main.main(
        checkpoint=False,
        epochs=4096,
        
        data=dataset,
        model=model,

        n_actions=2,
        
        # optuna params for GRU
        embedding_size=22,
        n_steps=32,
        learning_rate=0.00023,
        
        batch_size=-1,
        sequence_length=-1,
        train_test_ratio=0.5,
        scheduler=True,
        bagging=True,
        )

    losses.append(loss)

print(losses)