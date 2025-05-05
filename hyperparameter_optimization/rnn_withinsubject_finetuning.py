import sys
import os
import traceback
import logging
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optuna
import glob
import json
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resources.model_evaluation import bayesian_information_criterion, log_likelihood
from resources.bandits import BanditsDrift, AgentQ, AgentNetwork, AgentSpice, get_update_dynamics
from resources.rnn import RLRNN
from resources.rnn_utils import DatasetRNN
from resources.rnn_training import fit_model
from resources.sindy_training import fit_spice

np.random.seed(42)
torch.manual_seed(42)

start_time = time.time()

# QUICK CONFIG
epochs_rnn = 4096
scheduler = True
n_trials_optuna = 1#50

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

n_actions = 2

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def load_and_prepare_data(data_path):
    """
    Load and prepare data from CSV file.
    """
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    unique_participants = df['session'].unique()
    n_participants = len(unique_participants)
    logger.info(f"Number of participants: {n_participants}")

    all_xs = []
    all_ys = []

    for i, participant_id in enumerate(unique_participants):
        participant_df = df[df['session'] == participant_id]
        n_trials = len(participant_df)
        
        alpha_reward = participant_df['alpha_reward'].iloc[0]
        alpha_penalty = participant_df['alpha_penalty'].iloc[0] if 'alpha_penalty' in participant_df.columns else alpha_reward * 0.5
        
        xs = torch.zeros((1, n_trials, 5))
        for t in range(1, n_trials):
            prev_choice = participant_df['choice'].iloc[t-1]
            xs[0, t, int(prev_choice)] = 1.0
            if int(prev_choice) == 0:
                xs[0, t, 2] = participant_df['reward'].iloc[t-1]
                xs[0, t, 3] = -1
            else:
                xs[0, t, 2] = -1
                xs[0, t, 3] = participant_df['reward'].iloc[t-1]
        xs[0, :, 4] = i
        
        ys = torch.zeros((1, n_trials, n_actions))
        for t in range(n_trials):
            choice = participant_df['choice'].iloc[t]
            ys[0, t, int(choice)] = 1.0
        
        all_xs.append(xs)
        all_ys.append(ys)
        logger.info(f"Participant {i} (ID={participant_id}): α_reward={alpha_reward:.2f}, α_penalty={alpha_penalty:.2f}")

    for i, xs in enumerate(all_xs):
        logger.info(f"Participant {i} xs shape: {xs.shape}")
    
    combined_xs = torch.cat(all_xs)
    combined_ys = torch.cat(all_ys)
    
    logger.info(f"Combined xs shape after concatenation: {combined_xs.shape}")
    logger.info(f"Combined ys shape after concatenation: {combined_ys.shape}")
    
    combined_dataset = DatasetRNN(combined_xs, combined_ys)
    
    logger.info(f"Combined dataset shape: X={combined_dataset.xs.shape}, Y={combined_dataset.ys.shape}")
    
    return combined_dataset, n_participants

def split_dataset_within_participants(dataset, train_ratio=0.75):
    """
    Split dataset by trials within each participant.
    For each participant, use train_ratio% of trials for training and the rest for validation.
    70/30 train/test split by default.
    """
    all_participant_ids = torch.unique(dataset.xs[:, 0, -1]).tolist()
    n_participants = len(all_participant_ids)
    
    logger.info(f"Total unique participants: {n_participants}")
    logger.info(f"Train/test split ratio: {train_ratio}/{1-train_ratio} of trials within each participant")
    
    train_xs_list = []
    train_ys_list = []
    val_xs_list = []
    val_ys_list = []
    
    # For each participant, split their trials
    for pid in all_participant_ids:
        participant_mask = (dataset.xs[:, 0, -1] == pid)
        participant_xs = dataset.xs[participant_mask]
        participant_ys = dataset.ys[participant_mask]
        
        # Get sequence length (number of trials)
        seq_length = participant_xs.shape[1]
        
        # Compute split index
        split_idx = int(seq_length * train_ratio)
        
        # Create training data with first split_idx trials
        train_participant_xs = participant_xs[:, :split_idx, :]
        train_participant_ys = participant_ys[:, :split_idx, :]
        
        # Create validation data with remaining trials
        val_participant_xs = participant_xs[:, split_idx:, :]
        val_participant_ys = participant_ys[:, split_idx:, :]
        
        train_xs_list.append(train_participant_xs)
        train_ys_list.append(train_participant_ys)
        val_xs_list.append(val_participant_xs)
        val_ys_list.append(val_participant_ys)
        
        logger.info(f"Participant {pid}: {split_idx} trials for training, {seq_length - split_idx} trials for validation")
    
    train_xs = torch.cat(train_xs_list)
    train_ys = torch.cat(train_ys_list)
    val_xs = torch.cat(val_xs_list)
    val_ys = torch.cat(val_ys_list)
    
    logger.info(f"Train xs shape: {train_xs.shape}")
    logger.info(f"Train ys shape: {train_ys.shape}")
    logger.info(f"Validation xs shape: {val_xs.shape}")
    logger.info(f"Validation ys shape: {val_ys.shape}")
    
    train_dataset = DatasetRNN(train_xs, train_ys)
    val_dataset = DatasetRNN(val_xs, val_ys)
    
    return train_dataset, val_dataset, all_participant_ids

def define_sindy_configuration():
    """
    Define configuration for SINDy model.
    """
    rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
    control_parameters = ['c_action', 'c_reward', 'c_value_reward', 'c_value_choice']
    sindy_library_setup = {
        'x_learning_rate_reward': ['c_reward', 'c_value_reward', 'c_value_choice'],
        'x_value_reward_not_chosen': ['c_value_choice'],
        'x_value_choice_chosen': ['c_value_reward'],
        'x_value_choice_not_chosen': ['c_value_reward'],
    }
    sindy_filter_setup = {
        'x_learning_rate_reward': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }
    # sindy_dataprocessing = {
    #     'x_learning_rate_reward': [0, 0, 0],
    #     'x_value_reward_not_chosen': [0, 0, 0],
    #     'x_value_choice_chosen': [1, 1, 0],
    #     'x_value_choice_not_chosen': [1, 1, 0],
    #     'c_value_reward': [0, 0, 0],
    #     'c_value_choice': [1, 1, 0],
    # }
    return rnn_modules, control_parameters, sindy_library_setup, sindy_filter_setup

def objective(trial, train_dataset, val_dataset, n_participants):
    """
    Optuna objective function for hyperparameter optimization.
    """
    list_rnn_modules, list_control_parameters, library_setup, filter_setup = define_sindy_configuration()
    
    # embedding_size = trial.suggest_int('embedding_size', 8, 32)
    # learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    l1_weight_decay = trial.suggest_float('l1_weight_decay', 1e-6, 1e-2, log=True)
    l2_weight_decay = trial.suggest_float('l2_weight_decay', 1e-6, 1e-2, log=True)  
    # n_steps = trial.suggest_int('n_steps', 1, 100)
    # sindy_optimizer_alpha = trial.suggest_float('sindy_optimizer_alpha', 1e-2, 1, log=True)
    # sindy_optimizer_threshold = trial.suggest_float('sindy_optimizer_threshold', 1e-3, 1e-1, log=True)
    
    # logger.info(f"Trial {trial.number}: lr={learning_rate:.6f}, embedding_size={embedding_size}, n_steps={n_steps}, l1_weight_decay={l1_weight_decay:.6f}, l2_weight_decay={l2_weight_decay:.6f}, sindy_optimizer_alpha={sindy_optimizer_alpha:.6f}, sindy_optimizer_threshold={sindy_optimizer_threshold:.6f}")
    logger.info(f"Trial {trial.number}: l1_weight_decay={l1_weight_decay:.6f}, l2_weight_decay={l2_weight_decay:.6f}")
    
    model_rnn = RLRNN(
        n_actions=n_actions,
        n_participants=n_participants,
        hidden_size=8,
        embedding_size=32,
        dropout=0,
        l1_weight_decay=l1_weight_decay,
        l2_weight_decay=l2_weight_decay,
        list_signals=list_rnn_modules + list_control_parameters
    )
    
    optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=1e-3)
    
    try:
        model_rnn, optimizer_rnn, final_train_loss_rnn = fit_model(
            model=model_rnn,
            optimizer=optimizer_rnn,
            dataset_train=train_dataset,
            dataset_test=val_dataset,
            epochs=epochs_rnn,
            n_steps=16,
            scheduler=scheduler,
            convergence_threshold=0,
        )
        
        agent_rnn = AgentNetwork(model_rnn=model_rnn, n_actions=n_actions)
        
        agent_spice, final_train_loss_spice = fit_spice(
            agent=agent_rnn, 
            data=train_dataset, 
            get_loss=True,
            rnn_modules=list_rnn_modules,
            control_signals=list_control_parameters,
            library_setup=library_setup,
            filter_setup=filter_setup,
            optimizer_alpha=0.1,
            optimizer_threshold=0.05,
            )
        n_parameters_spice = agent_spice.count_parameters(mapping_modules_values={module: 'x_value_choice' if 'choice' in module else 'x_value_reward' for module in agent_spice._model.submodules_sindy})
        
        logger.info(f"Trial {trial.number}: RNN Train Loss: {final_train_loss_rnn:.7f}; SPICE Train Loss: {final_train_loss_spice}")
        
        val_loss = 0.0
        n_trials_eval = 0
        
        val_participant_ids = torch.unique(val_dataset.xs[:, 0, -1]).tolist()
        logger.info(f"Trial {trial.number}: Validation set has {len(val_participant_ids)} participants")
        
        for pid in val_participant_ids:
            mask = (val_dataset.xs[:, 0, -1] == pid)
            
            xs_val = val_dataset.xs[mask]
            ys_val = val_dataset.ys[mask]
            
            agent_spice.new_sess(participant_id=pid)
            
            # Get dynamics (choices and probabilities)
            _, probs, _ = get_update_dynamics(xs_val[0], agent_rnn)
            choices_np = ys_val[0, :len(probs)].cpu().numpy()
            
            # Calculate negative log likelihood
            # loss = -np.mean(np.sum(choices_np * np.log(probs + 1e-10), axis=1))
            loss = bayesian_information_criterion(data=choices_np, probs=probs, n_parameters=n_parameters_spice[pid])
            
            val_loss += loss
            n_trials_eval += len(probs)
        
        avg_val_loss = val_loss / n_trials_eval if n_trials_eval > 0 else float('inf')
        logger.info(f"Trial {trial.number}: Average Validation Loss: {avg_val_loss:.4f}, Eval count: {n_trials_eval}")
        
        return avg_val_loss
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        logger.error(traceback.format_exc())
        return float('inf')

def evaluate_with_sindy(model_rnn, val_dataset, participant_ids, n_participants, best_params):
    """
    Evaluate using SINDy by fitting separate models for each participant's validation trials.
    """
    list_rnn_modules, list_control_parameters, library_setup, filter_setup = define_sindy_configuration()
    
    # Create agent from RNN
    agent_rnn = AgentNetwork(model_rnn=model_rnn, n_actions=n_actions)
    
    logger.info("Evaluating with SINDy - fitting separate models for each participant's validation trials")
    
    all_bic_values = []
    all_ll_values = []
    participant_equations = {}
    participant_metrics = {}
    
    # Process each participant
    for pid in participant_ids:
        logger.info(f"Processing participant {pid}...")
        
        mask = (val_dataset.xs[:, 0, -1] == pid)
        xs_val = val_dataset.xs[mask]
        ys_val = val_dataset.ys[mask]

        participant_dataset = DatasetRNN(xs_val, ys_val)
        
        agent_rnn.new_sess(participant_id=pid)
        
        try:
            logger.info(f"Fitting SINDy model for participant {pid}")
            
            # Create SINDy agent just for this participant
            participant_sindy, _ = fit_spice(
                rnn_modules=list_rnn_modules,
                control_signals=list_control_parameters,
                agent=agent_rnn,
                data=participant_dataset,
                library_setup=library_setup,
                filter_setup=filter_setup,
                verbose=True,
                n_sessions_off_policy=0,
                optimizer_alpha=best_params['sindy_optimizer_alpha'],
                optimizer_threshold=best_params['sindy_optimizer_threshold'],
            )
            
            participant_equations[pid] = {}
            
            # Extract equations from SINDy
            for module_name in list_rnn_modules:
                if (hasattr(participant_sindy._model, 'submodules_sindy') and 
                    module_name in participant_sindy._model.submodules_sindy and 
                    pid in participant_sindy._model.submodules_sindy[module_name]):
                    
                    sindy_model = participant_sindy._model.submodules_sindy[module_name][pid]
                    equation_str = str(sindy_model)
                    coeffs = sindy_model.coefficients() if hasattr(sindy_model, 'coefficients') else None
                    
                    participant_equations[pid][module_name] = {
                        'equation': equation_str,
                        'coefficients': coeffs[0].tolist() if coeffs is not None and len(coeffs) > 0 else None,
                        'feature_names': sindy_model.feature_names if hasattr(sindy_model, 'feature_names') else None
                    }
            
            # Get SINDy parameter count
            participant_sindy.new_sess(participant_id=pid)
            sindy_params = participant_sindy.count_parameters()[pid] if hasattr(participant_sindy, 'count_parameters') else 0
            
            # Evaluate SINDy on validation trials
            _, probs, _ = get_update_dynamics(xs_val[0], participant_sindy)
            choices_np = ys_val[0].cpu().numpy()
            
            # Calculate log likelihood
            ll = log_likelihood(data=choices_np, probs=probs)
            n_trials = choices_np.shape[0]
            normalized_ll = ll / n_trials
            
            # Calculate BIC
            bic = bayesian_information_criterion(
                data=choices_np, 
                probs=probs, 
                n_parameters=sindy_params, 
                ll=ll
            )
            normalized_bic = bic / n_trials
            
            participant_metrics[str(pid)] = {
                "ll": float(normalized_ll),
                "bic": float(normalized_bic),
                "params": int(sindy_params),
                "n_val_trials": int(n_trials)
            }
            
            logger.info(f"Participant {pid}: LL={normalized_ll:.4f}, BIC={normalized_bic:.4f}, Params={sindy_params}, Val trials={n_trials}")
            
            all_bic_values.append(normalized_bic)
            all_ll_values.append(normalized_ll)
            
        except Exception as e:
            logger.warning(f"Error processing participant {pid}: {str(e)}")
            logger.warning(traceback.format_exc())
    
    # Calculate average metrics
    avg_bic = np.mean(all_bic_values) if all_bic_values else float('nan')
    avg_ll = np.mean(all_ll_values) if all_ll_values else float('nan')
    
    logger.info(f"Number of participants with valid BIC metrics: {len(all_bic_values)}/{len(participant_ids)}")
    logger.info(f"Average SINDy BIC: {avg_bic:.4f}")
    logger.info(f"Average SINDy LL: {avg_ll:.4f}")
    
    return avg_bic, avg_ll, participant_equations, participant_metrics

def create_violin_plots(output_dir, all_results):
    """
    Violin plots with scatter points for BIC and LL values.
    """
    ll_data = []
    bic_data = []
    
    for result in all_results:
        dataset_name = result["dataset"].replace('.csv', '').replace('data_rldm_', '')
        
        if "participant_metrics" in result:
            for pid, metrics in result["participant_metrics"].items():
                ll_data.append({
                    "dataset": dataset_name,
                    "participant": pid,
                    "value": metrics["ll"]
                })
                
                bic_data.append({
                    "dataset": dataset_name,
                    "participant": pid,
                    "value": metrics["bic"]
                })
    
    if ll_data:
        ll_df = pd.DataFrame(ll_data)
        bic_df = pd.DataFrame(bic_data)
        
        plt.figure(figsize=(12, 8))
        ax = sns.violinplot(x="dataset", y="value", data=ll_df, inner="quartile", palette="Set2")
        sns.stripplot(x="dataset", y="value", data=ll_df, size=5, color="black", alpha=0.5, ax=ax)
        
        plt.xlabel('Datasets', fontsize=14)
        plt.ylabel('Normalized Log Likelihood', fontsize=14)
        plt.title('Log Likelihood Distribution by Dataset', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, dataset in enumerate(ll_df["dataset"].unique()):
            mean_val = ll_df[ll_df["dataset"] == dataset]["value"].mean()
            median_val = ll_df[ll_df["dataset"] == dataset]["value"].median()
            plt.text(i, ll_df["value"].min() - 0.1, 
                    f'Mean: {mean_val:.4f}\nMedian: {median_val:.4f}', 
                    ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'll_violin_plot.png'), dpi=300)
        
        #  BIC violin plot
        plt.figure(figsize=(12, 8))
        ax = sns.violinplot(x="dataset", y="value", data=bic_df, inner="quartile", palette="Set2")
        sns.stripplot(x="dataset", y="value", data=bic_df, size=5, color="black", alpha=0.5, ax=ax)
        
        plt.xlabel('Datasets', fontsize=14)
        plt.ylabel('Normalized BIC', fontsize=14)
        plt.title('BIC Distribution by Dataset', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, dataset in enumerate(bic_df["dataset"].unique()):
            mean_val = bic_df[bic_df["dataset"] == dataset]["value"].mean()
            median_val = bic_df[bic_df["dataset"] == dataset]["value"].median()
            plt.text(i, bic_df["value"].max() + 0.1, 
                    f'Mean: {mean_val:.4f}\nMedian: {median_val:.4f}', 
                    ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bic_violin_plot.png'), dpi=300)
        
        logger.info(f"Created violin plots with {len(ll_df)} participant data points")
    else:
        logger.warning("No participant-level metrics found for violin plots")

def main():

    logger.info("=" * 80)
    logger.info("EXPERIMENT CONFIG")
    logger.info("=" * 80)
    
    # data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/")
    data_dir = "data/optuna"
    # all_data_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    all_data_files = os.listdir(data_dir)
    
    logger.info(f"Found {len(all_data_files)} data files: {[os.path.basename(f) for f in all_data_files]}")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning_within_subj_esults")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # for data_file in all_data_files:
    for data_file in os.listdir(data_dir):
        data_file = os.path.join(data_dir, data_file)
        dataset_name = os.path.basename(data_file)
        logger.info(f"Processing dataset: {dataset_name}")
        
        combined_dataset, n_participants = load_and_prepare_data(data_file)
        
        # Split trials within each participant
        train_dataset, val_dataset, participant_ids = split_dataset_within_participants(combined_dataset, train_ratio=0.7)
        
        logger.info(f"Train dataset: {train_dataset.xs.shape}, Validation dataset: {val_dataset.xs.shape}")
        
        study = optuna.create_study(direction="minimize")
        
        objective_func = lambda trial: objective(trial, train_dataset, val_dataset, n_participants)
        
        logger.info("Starting hyperparameter optimization...")
        study.optimize(objective_func, n_trials=n_trials_optuna)
        
        best_params = study.best_params
        best_value = study.best_value
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation loss: {best_value:.4f}")
        
        # Create the best RNN model
        model_rnn = RLRNN(
            n_actions=n_actions,
            n_participants=n_participants,
            # hidden_size=best_params['hidden_size'],
            embedding_size=best_params['embedding_size'],
            # dropout=best_params['dropout'],
            list_signals=define_sindy_configuration()[0] + define_sindy_configuration()[1]
        )
        
        optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=best_params['learning_rate'])
        
        # Train the model with best parameters
        model_rnn, optimizer_rnn, final_train_loss = fit_model(
            model=model_rnn,
            optimizer=optimizer_rnn,
            dataset_train=train_dataset,
            dataset_test=val_dataset,
            epochs=epochs_rnn,  
            n_steps=best_params['n_steps'],
            scheduler=scheduler,
            convergence_threshold=1e-14,
        )
        
        logger.info(f"Final RNN training loss: {final_train_loss:.7f}")
        
        # Evaluate with SINDy to get BIC and LL
        avg_bic, avg_ll, participant_equations, participant_metrics = evaluate_with_sindy(
            model_rnn, val_dataset, participant_ids, n_participants, best_params
        )
        
        result = {
            "dataset": dataset_name,
            "best_params": best_params,
            "best_val_loss": float(best_value),
            "final_train_loss": float(final_train_loss),
            "sindy_bic": float(avg_bic),
            "sindy_ll": float(avg_ll),
            "participant_equations": participant_equations,
            "participant_metrics": participant_metrics  
        }
        
        all_results.append(result)
        
        result_file = os.path.join(output_dir, f"result_{dataset_name.replace('.csv', '')}.json")
        with open(result_file, 'w') as f:
            serializable_result = convert_numpy_types(result)
            json.dump(serializable_result, f, indent=2)
        
        model_file = os.path.join(output_dir, f"model_{dataset_name.replace('.csv', '')}.pt")
        torch.save(model_rnn.state_dict(), model_file)
        
        logger.info(f"Completed processing dataset: {dataset_name}")
    
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        serializable_results = convert_numpy_types({"results": all_results})
        json.dump(serializable_results, f, indent=2)
    
    try:
        dataset_names = [r["dataset"].replace('.csv', '').replace('data_rldm_', '') for r in all_results]
        bic_values = [r["sindy_bic"] for r in all_results]
        ll_values = [r["sindy_ll"] for r in all_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(dataset_names, bic_values)
        plt.xlabel('Datasets')
        plt.ylabel('Average Normalized BIC')
        plt.title('SINDy BIC Comparison Across Datasets')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sindy_bic_comparison.png'))
        
        plt.figure(figsize=(10, 6))
        plt.bar(dataset_names, ll_values)
        plt.xlabel('Datasets')
        plt.ylabel('Average Normalized Log Likelihood')
        plt.title('Log Likelihood Comparison Across Datasets')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'll_comparison.png'))
        
        create_violin_plots(output_dir, all_results)
        
    except Exception as e:
        logger.error(f"Error generating comparison plots: {str(e)}")
        logger.error(traceback.format_exc())
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time: {end_time - start_time} seconds")