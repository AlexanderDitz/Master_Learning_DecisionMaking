import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn

path_datasets = 'data/parameter_recovery/'
path_params = 'params/parameter_recovery/'

datasets = os.listdir(path_datasets)

losses = []
for d in datasets:
    dataset = os.path.join(path_datasets, d)
    model = os.path.join(path_params, d.replace('.csv', f'.pkl').replace('data', 'rnn'))
    
    _, loss = pipeline_rnn.main(
        checkpoint=False,
        epochs=16,#4096,
        
        data=dataset,
        model=model,

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
        embedding_size=22,
        train_test_ratio=0.5,
        scheduler=False,
        
        batch_size=-1,
        sequence_length=-1,
        bagging=True,
        )

    losses.append(loss)

print(losses)