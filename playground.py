import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_mcmc
from benchmarking.lstm_training import setup_agent_lstm
from utils.plotting import plot_session
from resources.rnn import RLRNN_dezfouli2019, RLRNN_eckstein2022
from resources.sindy_utils import SindyConfig_dezfouli2019, SindyConfig_eckstein2022
from resources.bandits import AgentQ
from resources.rnn_utils import split_data_along_sessiondim
from utils.convert_dataset import convert_dataset

# Your existing code
path_data = 'data/eckstein2022/eckstein2022.csv'
path_rnn = 'params/eckstein2022/rnn_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'
agent_rnn = setup_agent_rnn(
    class_rnn=RLRNN_eckstein2022,
    path_model=path_rnn,
    list_sindy_signals=SindyConfig_eckstein2022['rnn_modules']+SindyConfig_eckstein2022['control_parameters'],
)
dataset = convert_dataset(path_data, additional_inputs=['age'])[0]
age = dataset.xs[:, 0, -4]

# Get all embeddings from RNN
embeddings = torch.nn.functional.leaky_relu(
    agent_rnn._model.participant_embedding(torch.arange(len(dataset))), 
    negative_slope=0.001
).detach().numpy()

# Perform t-SNE on embeddings
embeddings_reduced = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

plt.scatter(
    embeddings_reduced[:, 0], 
    embeddings_reduced[:, 1],
    alpha=0.7,
    s=50
)
plt.show()
# plt.savefig('scatter_embedding', dpi=500)
# # Define age limits and create age groups
# age_limits = np.array([8, 10, 13, 15, 18, 21, 25, 30]) / 30  # Normalized age limits
# age_groups = np.digitize(age, age_limits)

# # Create age group labels
# age_group_labels = [
#     f'{int(age_limits[i]*30)}-{int(age_limits[i+1]*30)}' if i < len(age_limits)-1 
#     else f'{int(age_limits[i]*30)}+' 
#     for i in range(len(age_limits))
# ]
# age_group_labels = ['<8'] + age_group_labels

# # Create a colormap for age groups
# n_groups = len(np.unique(age_groups))
# colors = plt.cm.tab10(np.linspace(0, 1, n_groups))

# # Create the plot with subplots for marginal densities
# fig = plt.figure(figsize=(12, 10))
# gs = fig.add_gridspec(3, 3, width_ratios=[1, 3, 0.3], height_ratios=[1, 3, 0.3],
#                       hspace=0.05, wspace=0.05)

# # Main scatter plot
# ax_main = fig.add_subplot(gs[1, 1])
# ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
# ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)

# Plot the main scatter plot
# for group_idx in np.unique(age_groups):
#     mask = age_groups == group_idx
#     ax_main.scatter(
#         embeddings_reduced[mask, 0], 
#         embeddings_reduced[mask, 1],
#         c=[colors[group_idx]], 
#         label=age_group_labels[group_idx],
#         alpha=0.7,
#         s=50
#     )

# ax_main.set_xlabel('t-SNE Dimension 1', fontsize=12)
# ax_main.set_ylabel('t-SNE Dimension 2', fontsize=12)
# ax_main.legend(title='Age Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
# ax_main.grid(True, alpha=0.3)

# # Plot marginal densities for x-axis (top)
# x_range = np.linspace(embeddings_reduced[:, 0].min(), embeddings_reduced[:, 0].max(), 200)
# for group_idx in np.unique(age_groups):
#     mask = age_groups == group_idx
#     if np.sum(mask) > 1:  # Need at least 2 points for KDE
#         kde = gaussian_kde(embeddings_reduced[mask, 0])
#         density = kde(x_range)
#         ax_top.fill_between(x_range, density, alpha=0.6, color=colors[group_idx], 
#                            label=age_group_labels[group_idx])

# ax_top.set_ylabel('Density', fontsize=10)
# ax_top.tick_params(labelbottom=False)
# ax_top.grid(True, alpha=0.3)

# # Plot marginal densities for y-axis (right)
# y_range = np.linspace(embeddings_reduced[:, 1].min(), embeddings_reduced[:, 1].max(), 200)
# for group_idx in np.unique(age_groups):
#     mask = age_groups == group_idx
#     if np.sum(mask) > 1:  # Need at least 2 points for KDE
#         kde = gaussian_kde(embeddings_reduced[mask, 1])
#         density = kde(y_range)
#         ax_right.fill_betweenx(y_range, density, alpha=0.6, color=colors[group_idx], 
#                               label=age_group_labels[group_idx])

# ax_right.set_xlabel('Density', fontsize=10)
# ax_right.tick_params(labelleft=False)
# ax_right.grid(True, alpha=0.3)

# # Set title
# fig.suptitle('t-SNE Visualization of Participant Embeddings by Age Group', 
#              fontsize=14, fontweight='bold')

# plt.tight_layout()
# plt.show()

# # Print some statistics
# print(f"Total participants: {len(age)}")
# print(f"Age range: {age.min():.2f} - {age.max():.2f}")
# print(f"Number of age groups: {n_groups}")
# print(f"t-SNE embedding shape: {embeddings_reduced.shape}")

# # Print age group distribution
# for group_idx in np.unique(age_groups):
#     count = np.sum(age_groups == group_idx)
#     print(f"Age group '{age_group_labels[group_idx]}': {count} participants")

# # ============================================================================
# # SECOND PLOT: Grouped by Accumulated Reward Quartiles
# # ============================================================================

# # Calculate accumulated reward per participant
# # Assuming reward is in the dataset - you may need to adjust this based on your data structure
# # This is a placeholder - replace with your actual reward calculation
# accumulated_rewards = np.random.rand(len(age)) * 100  # Replace this line with actual reward calculation
# # Example: accumulated_rewards = dataset.rewards.sum(axis=1)  # Adjust based on your data structure

# # Calculate quartiles
# q25 = np.percentile(accumulated_rewards, 25)
# q75 = np.percentile(accumulated_rewards, 75)

# # Create reward groups: 0=lower quartile, 1=middle 50%, 2=upper quartile
# reward_groups = np.zeros(len(accumulated_rewards), dtype=int)
# reward_groups[accumulated_rewards <= q25] = 0  # Lower quartile
# reward_groups[(accumulated_rewards > q25) & (accumulated_rewards < q75)] = 1  # Middle 50%
# reward_groups[accumulated_rewards >= q75] = 2  # Upper quartile

# # Create reward group labels
# reward_group_labels = [
#     f'Lower Quartile (≤{q25:.1f})',
#     f'Middle 50% ({q25:.1f}-{q75:.1f})',
#     f'Upper Quartile (≥{q75:.1f})'
# ]

# # Create colors for reward groups
# reward_colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green

# # Create the second plot
# fig2 = plt.figure(figsize=(12, 10))
# gs2 = fig2.add_gridspec(3, 3, width_ratios=[1, 3, 0.3], height_ratios=[1, 3, 0.3],
#                        hspace=0.05, wspace=0.05)

# # Main scatter plot for rewards
# ax_main2 = fig2.add_subplot(gs2[1, 1])
# ax_top2 = fig2.add_subplot(gs2[0, 1], sharex=ax_main2)
# ax_right2 = fig2.add_subplot(gs2[1, 2], sharey=ax_main2)

# # Plot the main scatter plot for reward groups
# for group_idx in range(3):
#     mask = reward_groups == group_idx
#     ax_main2.scatter(
#         embeddings_reduced[mask, 0], 
#         embeddings_reduced[mask, 1],
#         c=reward_colors[group_idx], 
#         label=reward_group_labels[group_idx],
#         alpha=0.7,
#         s=50
#     )

# ax_main2.set_xlabel('t-SNE Dimension 1', fontsize=12)
# ax_main2.set_ylabel('t-SNE Dimension 2', fontsize=12)
# ax_main2.legend(title='Reward Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
# ax_main2.grid(True, alpha=0.3)

# # Plot marginal densities for x-axis (top) - reward groups
# x_range = np.linspace(embeddings_reduced[:, 0].min(), embeddings_reduced[:, 0].max(), 200)
# for group_idx in range(3):
#     mask = reward_groups == group_idx
#     if np.sum(mask) > 1:  # Need at least 2 points for KDE
#         kde = gaussian_kde(embeddings_reduced[mask, 0])
#         density = kde(x_range)
#         ax_top2.fill_between(x_range, density, alpha=0.6, color=reward_colors[group_idx], 
#                            label=reward_group_labels[group_idx])

# ax_top2.set_ylabel('Density', fontsize=10)
# ax_top2.tick_params(labelbottom=False)
# ax_top2.grid(True, alpha=0.3)

# # Plot marginal densities for y-axis (right) - reward groups
# y_range = np.linspace(embeddings_reduced[:, 1].min(), embeddings_reduced[:, 1].max(), 200)
# for group_idx in range(3):
#     mask = reward_groups == group_idx
#     if np.sum(mask) > 1:  # Need at least 2 points for KDE
#         kde = gaussian_kde(embeddings_reduced[mask, 1])
#         density = kde(y_range)
#         ax_right2.fill_betweenx(y_range, density, alpha=0.6, color=reward_colors[group_idx], 
#                               label=reward_group_labels[group_idx])

# ax_right2.set_xlabel('Density', fontsize=10)
# ax_right2.tick_params(labelleft=False)
# ax_right2.grid(True, alpha=0.3)

# # Set title
# fig2.suptitle('t-SNE Visualization of Participant Embeddings by Accumulated Reward', 
#              fontsize=14, fontweight='bold')

# plt.tight_layout()
# plt.show()

# # Print reward statistics
# print(f"\n--- Reward Analysis ---")
# print(f"Accumulated reward range: {accumulated_rewards.min():.2f} - {accumulated_rewards.max():.2f}")
# print(f"25th percentile (Q1): {q25:.2f}")
# print(f"75th percentile (Q3): {q75:.2f}")
# print(f"Median: {np.percentile(accumulated_rewards, 50):.2f}")

# # Print reward group distribution
# for group_idx in range(3):
#     count = np.sum(reward_groups == group_idx)
#     avg_reward = accumulated_rewards[reward_groups == group_idx].mean()
#     print(f"Reward group '{reward_group_labels[group_idx]}': {count} participants (avg: {avg_reward:.2f})")

# # ============================================================================
# # BEHAVIORAL METRICS CALCULATION
# # ============================================================================

# def calculate_behavioral_metrics(dataset):
#     """Calculate behavioral metrics for each participant"""
#     n_participants = len(dataset)
    
#     # Initialize metrics
#     switch_rates = np.zeros(n_participants)
#     stay_after_reward_rates = np.zeros(n_participants)
#     perseveration_rates = np.zeros(n_participants)
    
#     for p in range(n_participants):
#         # Get participant data (assuming actions and rewards are in dataset)
#         # You may need to adjust these based on your dataset structure
#         actions = dataset.actions[p]  # Replace with actual action data access
#         rewards = dataset.rewards[p]  # Replace with actual reward data access
        
#         # Calculate switch rate (proportion of action switches)
#         switches = np.sum(actions[1:] != actions[:-1])
#         switch_rates[p] = switches / (len(actions) - 1) if len(actions) > 1 else 0
        
#         # Calculate stay after reward rate
#         rewarded_trials = np.where(rewards[:-1] > 0)[0]  # Trials with reward (excluding last)
#         if len(rewarded_trials) > 0:
#             stays_after_reward = np.sum(actions[rewarded_trials] == actions[rewarded_trials + 1])
#             stay_after_reward_rates[p] = stays_after_reward / len(rewarded_trials)
#         else:
#             stay_after_reward_rates[p] = 0
        
#         # Calculate perseveration (staying with same action after unrewarded trials)
#         unrewarded_trials = np.where(rewards[:-1] == 0)[0]  # Trials without reward (excluding last)
#         if len(unrewarded_trials) > 0:
#             perseverations = np.sum(actions[unrewarded_trials] == actions[unrewarded_trials + 1])
#             perseveration_rates[p] = perseverations / len(unrewarded_trials)
#         else:
#             perseveration_rates[p] = 0
    
#     return switch_rates, stay_after_reward_rates, perseveration_rates

# # Calculate behavioral metrics (replace with your actual calculation)
# # This is a placeholder - replace with actual behavioral metric calculation
# switch_rates = np.random.rand(len(age)) * 0.5 + 0.25  # Replace this line
# stay_after_reward_rates = np.random.rand(len(age)) * 0.6 + 0.4  # Replace this line  
# perseveration_rates = np.random.rand(len(age)) * 0.4 + 0.1  # Replace this line

# # Uncomment and use this when you have the actual data:
# # switch_rates, stay_after_reward_rates, perseveration_rates = calculate_behavioral_metrics(dataset)

# def create_behavioral_plot(metric_values, metric_name, colors_scheme='RdYlGn'):
#     """Create a plot for a behavioral metric grouped by quartiles"""
    
#     # Calculate quartiles
#     q25 = np.percentile(metric_values, 25)
#     q75 = np.percentile(metric_values, 75)
    
#     # Create groups: 0=lower quartile, 1=middle 50%, 2=upper quartile
#     groups = np.zeros(len(metric_values), dtype=int)
#     groups[metric_values <= q25] = 0
#     groups[(metric_values > q25) & (metric_values < q75)] = 1
#     groups[metric_values >= q75] = 2
    
#     # Create group labels
#     group_labels = [
#         f'Lower Quartile (≤{q25:.2f})',
#         f'Middle 50% ({q25:.2f}-{q75:.2f})',
#         f'Upper Quartile (≥{q75:.2f})'
#     ]
    
#     # Set colors based on metric (reverse for switch rate since lower might be better)
#     if colors_scheme == 'RdYlGn':
#         group_colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green
#     elif colors_scheme == 'RdYlGn_r':
#         group_colors = ['#27ae60', '#f39c12', '#e74c3c']  # Green, Orange, Red
#     else:
#         group_colors = ['#3498db', '#9b59b6', '#e67e22']  # Blue, Purple, Orange
    
#     # Create the plot
#     fig = plt.figure(figsize=(12, 10))
#     gs = fig.add_gridspec(3, 3, width_ratios=[1, 3, 0.3], height_ratios=[1, 3, 0.3],
#                          hspace=0.05, wspace=0.05)
    
#     # Main scatter plot
#     ax_main = fig.add_subplot(gs[1, 1])
#     ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
#     ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    
#     # Plot the main scatter plot
#     for group_idx in range(3):
#         mask = groups == group_idx
#         ax_main.scatter(
#             embeddings_reduced[mask, 0], 
#             embeddings_reduced[mask, 1],
#             c=group_colors[group_idx], 
#             label=group_labels[group_idx],
#             alpha=0.7,
#             s=50
#         )
    
#     ax_main.set_xlabel('t-SNE Dimension 1', fontsize=12)
#     ax_main.set_ylabel('t-SNE Dimension 2', fontsize=12)
#     ax_main.legend(title=f'{metric_name} Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax_main.grid(True, alpha=0.3)
    
#     # Plot marginal densities for x-axis (top)
#     x_range = np.linspace(embeddings_reduced[:, 0].min(), embeddings_reduced[:, 0].max(), 200)
#     for group_idx in range(3):
#         mask = groups == group_idx
#         if np.sum(mask) > 1:
#             kde = gaussian_kde(embeddings_reduced[mask, 0])
#             density = kde(x_range)
#             ax_top.fill_between(x_range, density, alpha=0.6, color=group_colors[group_idx])
    
#     ax_top.set_ylabel('Density', fontsize=10)
#     ax_top.tick_params(labelbottom=False)
#     ax_top.grid(True, alpha=0.3)
    
#     # Plot marginal densities for y-axis (right)
#     y_range = np.linspace(embeddings_reduced[:, 1].min(), embeddings_reduced[:, 1].max(), 200)
#     for group_idx in range(3):
#         mask = groups == group_idx
#         if np.sum(mask) > 1:
#             kde = gaussian_kde(embeddings_reduced[mask, 1])
#             density = kde(y_range)
#             ax_right.fill_betweenx(y_range, density, alpha=0.6, color=group_colors[group_idx])
    
#     ax_right.set_xlabel('Density', fontsize=10)
#     ax_right.tick_params(labelleft=False)
#     ax_right.grid(True, alpha=0.3)
    
#     # Set title
#     fig.suptitle(f't-SNE Visualization of Participant Embeddings by {metric_name}', 
#                 fontsize=14, fontweight='bold')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print statistics
#     print(f"\n--- {metric_name} Analysis ---")
#     print(f"{metric_name} range: {metric_values.min():.3f} - {metric_values.max():.3f}")
#     print(f"25th percentile (Q1): {q25:.3f}")
#     print(f"75th percentile (Q3): {q75:.3f}")
#     print(f"Median: {np.percentile(metric_values, 50):.3f}")
    
#     for group_idx in range(3):
#         count = np.sum(groups == group_idx)
#         avg_value = metric_values[groups == group_idx].mean()
#         print(f"{metric_name} group '{group_labels[group_idx]}': {count} participants (avg: {avg_value:.3f})")

# # ============================================================================
# # THIRD PLOT: Switch Rate
# # ============================================================================
# create_behavioral_plot(switch_rates, 'Switch Rate', 'custom')

# # ============================================================================
# # FOURTH PLOT: Stay After Reward
# # ============================================================================
# create_behavioral_plot(stay_after_reward_rates, 'Stay After Reward', 'RdYlGn')

# # ============================================================================
# # FIFTH PLOT: Perseveration
# # ============================================================================
# create_behavioral_plot(perseveration_rates, 'Perseveration', 'RdYlGn_r')

# # ============================================================================
# # ANALYSIS FOR CATEGORICAL DIAGNOSIS DATASET
# # ============================================================================

# # Load the diagnosis dataset
# path_data_diagnosis = 'data/dezfouli2019/dezfouli2019.csv'  # Update with your actual path
# path_rnn_diagnosis = 'params/dezfouli2019/rnn_dezfouli2019_rldm_l1emb_0_001_l2_0_0001.pkl'  # Update with your actual path

# # Setup agent for diagnosis dataset (adjust if using different model)
# agent_rnn_diagnosis = setup_agent_rnn(
#     class_rnn=RLRNN_dezfouli2019,
#     path_model=path_rnn_diagnosis,  # Use diagnosis-specific model if available
#     list_sindy_signals=SindyConfig_dezfouli2019['rnn_modules']+SindyConfig_dezfouli2019['control_parameters'],
# )

# # Load diagnosis dataset
# dataset_diagnosis = convert_dataset(path_data_diagnosis, additional_inputs=['diag'])[0]
# diagnosis = dataset_diagnosis.xs[:, 0, -4].numpy()[::12]  # Adjust index based on your data structure
# diagnosis_types = np.unique(diagnosis)

# # Get embeddings for diagnosis dataset
# embeddings_diagnosis = torch.nn.functional.leaky_relu(
#     agent_rnn_diagnosis._model.participant_embedding(dataset_diagnosis.xs[:, 0, -1].int()[::12]), 
#     negative_slope=0.001
# ).detach().numpy()

# # Perform t-SNE on diagnosis embeddings
# embeddings_reduced_diagnosis = TSNE(n_components=2, random_state=42).fit_transform(embeddings_diagnosis)
# # embeddings_reduced_diagnosis = PCA(n_components=2, random_state=42).fit_transform(embeddings_diagnosis)


# def create_categorical_plot(embeddings_reduced, categories, category_labels, title_suffix, category_name):
#     """Create a plot for categorical variables"""
    
#     # Get unique categories
#     unique_categories = np.unique(categories)
#     n_categories = len(unique_categories)
    
#     # Create colors for categories
#     colors = plt.cm.Set1(np.linspace(0, 1, n_categories))
    
#     # Create the plot
#     fig = plt.figure(figsize=(12, 10))
#     gs = fig.add_gridspec(3, 3, width_ratios=[1, 3, 0.3], height_ratios=[1, 3, 0.3],
#                          hspace=0.05, wspace=0.05)
    
#     # Main scatter plot
#     ax_main = fig.add_subplot(gs[1, 1])
#     ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
#     ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    
#     # Plot the main scatter plot
#     for i, cat in enumerate(unique_categories):
#         mask = categories == cat
#         ax_main.scatter(
#             embeddings_reduced[mask, 0], 
#             embeddings_reduced[mask, 1],
#             c=[colors[i]], 
#             label=category_labels[int(cat)],
#             alpha=0.7,
#             s=50
#         )
    
#     ax_main.set_xlabel('t-SNE Dimension 1', fontsize=12)
#     ax_main.set_ylabel('t-SNE Dimension 2', fontsize=12)
#     ax_main.legend(title=f'{category_name}', bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax_main.grid(True, alpha=0.3)
    
#     # Plot marginal densities for x-axis (top)
#     x_range = np.linspace(embeddings_reduced[:, 0].min(), embeddings_reduced[:, 0].max(), 200)
#     for i, cat in enumerate(unique_categories):
#         mask = categories == cat
#         if np.sum(mask) > 1:
#             kde = gaussian_kde(embeddings_reduced[mask, 0])
#             density = kde(x_range)
#             ax_top.fill_between(x_range, density, alpha=0.6, color=colors[i], 
#                                label=category_labels[int(cat)])
    
#     ax_top.set_ylabel('Density', fontsize=10)
#     ax_top.tick_params(labelbottom=False)
#     ax_top.grid(True, alpha=0.3)
    
#     # Plot marginal densities for y-axis (right)
#     y_range = np.linspace(embeddings_reduced[:, 1].min(), embeddings_reduced[:, 1].max(), 200)
#     for i, cat in enumerate(unique_categories):
#         mask = categories == cat
#         if np.sum(mask) > 1:
#             kde = gaussian_kde(embeddings_reduced[mask, 1])
#             density = kde(y_range)
#             ax_right.fill_betweenx(y_range, density, alpha=0.6, color=colors[i], 
#                                   label=category_labels[int(cat)])
    
#     ax_right.set_xlabel('Density', fontsize=10)
#     ax_right.tick_params(labelleft=False)
#     ax_right.grid(True, alpha=0.3)
    
#     # Set title
#     fig.suptitle(f't-SNE Visualization of Participant Embeddings by {title_suffix}', 
#                 fontsize=14, fontweight='bold')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print statistics
#     print(f"\n--- {category_name} Analysis ---")
#     print(f"Total participants: {len(categories)}")
#     print(f"Categories: {unique_categories}")
    
#     for cat in unique_categories:
#         count = np.sum(categories == cat)
#         percentage = (count / len(categories)) * 100
#         print(f"{category_name} '{category_labels[int(cat)]}': {count} participants ({percentage:.1f}%)")

# # ============================================================================
# # DIAGNOSIS PLOTS
# # ============================================================================

# # Define diagnosis labels (adjust these based on your actual diagnosis coding)
# # diagnosis_labels = {
# #     0: 'Control/Healthy',
# #     1: 'Diagnosis Group 1', 
# #     2: 'Diagnosis Group 2'
# # }
# diagnosis_labels = {diagnosis_types[i]: ['H', 'D1', 'D2'][i] for i in range(len(diagnosis_types))}

# # Or more specific labels like:
# # diagnosis_labels = {
# #     0: 'Control',
# #     1: 'ADHD', 
# #     2: 'Depression'
# # }

# print("="*80)
# print("DIAGNOSIS DATASET ANALYSIS")
# print("="*80)

# # Main diagnosis plot
# create_categorical_plot(
#     embeddings_reduced_diagnosis, 
#     diagnosis, 
#     diagnosis_labels, 
#     'Diagnosis', 
#     'Diagnosis'
# )

# # Calculate behavioral metrics for diagnosis dataset (using placeholders for now)
# # Replace these with actual calculations from your diagnosis dataset
# switch_rates_diag = np.random.rand(len(diagnosis)) * 0.5 + 0.25
# stay_after_reward_rates_diag = np.random.rand(len(diagnosis)) * 0.6 + 0.4
# perseveration_rates_diag = np.random.rand(len(diagnosis)) * 0.4 + 0.1
# accumulated_rewards_diag = np.random.rand(len(diagnosis)) * 100

# # Uncomment when you have actual data:
# # switch_rates_diag, stay_after_reward_rates_diag, perseveration_rates_diag = calculate_behavioral_metrics(dataset_diagnosis)
# # accumulated_rewards_diag = calculate_accumulated_rewards(dataset_diagnosis)  # Implement this function

# def create_behavioral_plot_diagnosis(metric_values, metric_name, embeddings_reduced, colors_scheme='RdYlGn'):
#     """Create a plot for a behavioral metric grouped by quartiles for diagnosis dataset"""
    
#     # Calculate quartiles
#     q25 = np.percentile(metric_values, 25)
#     q75 = np.percentile(metric_values, 75)
    
#     # Create groups: 0=lower quartile, 1=middle 50%, 2=upper quartile
#     groups = np.zeros(len(metric_values), dtype=int)
#     groups[metric_values <= q25] = 0
#     groups[(metric_values > q25) & (metric_values < q75)] = 1
#     groups[metric_values >= q75] = 2
    
#     # Create group labels
#     group_labels = [
#         f'Lower Quartile (≤{q25:.2f})',
#         f'Middle 50% ({q25:.2f}-{q75:.2f})',
#         f'Upper Quartile (≥{q75:.2f})'
#     ]
    
#     # Set colors based on metric
#     if colors_scheme == 'RdYlGn':
#         group_colors = ['#e74c3c', '#f39c12', '#27ae60']
#     elif colors_scheme == 'RdYlGn_r':
#         group_colors = ['#27ae60', '#f39c12', '#e74c3c']
#     else:
#         group_colors = ['#3498db', '#9b59b6', '#e67e22']
    
#     # Create the plot
#     fig = plt.figure(figsize=(12, 10))
#     gs = fig.add_gridspec(3, 3, width_ratios=[1, 3, 0.3], height_ratios=[1, 3, 0.3],
#                          hspace=0.05, wspace=0.05)
    
#     # Main scatter plot
#     ax_main = fig.add_subplot(gs[1, 1])
#     ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
#     ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    
#     # Plot the main scatter plot
#     for group_idx in range(3):
#         mask = groups == group_idx
#         ax_main.scatter(
#             embeddings_reduced[mask, 0], 
#             embeddings_reduced[mask, 1],
#             c=group_colors[group_idx], 
#             label=group_labels[group_idx],
#             alpha=0.7,
#             s=50
#         )
    
#     ax_main.set_xlabel('t-SNE Dimension 1', fontsize=12)
#     ax_main.set_ylabel('t-SNE Dimension 2', fontsize=12)
#     ax_main.legend(title=f'{metric_name} Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax_main.grid(True, alpha=0.3)
    
#     # Plot marginal densities for x-axis (top)
#     x_range = np.linspace(embeddings_reduced[:, 0].min(), embeddings_reduced[:, 0].max(), 200)
#     for group_idx in range(3):
#         mask = groups == group_idx
#         if np.sum(mask) > 1:
#             kde = gaussian_kde(embeddings_reduced[mask, 0])
#             density = kde(x_range)
#             ax_top.fill_between(x_range, density, alpha=0.6, color=group_colors[group_idx])
    
#     ax_top.set_ylabel('Density', fontsize=10)
#     ax_top.tick_params(labelbottom=False)
#     ax_top.grid(True, alpha=0.3)
    
#     # Plot marginal densities for y-axis (right)
#     y_range = np.linspace(embeddings_reduced[:, 1].min(), embeddings_reduced[:, 1].max(), 200)
#     for group_idx in range(3):
#         mask = groups == group_idx
#         if np.sum(mask) > 1:
#             kde = gaussian_kde(embeddings_reduced[mask, 1])
#             density = kde(y_range)
#             ax_right.fill_betweenx(y_range, density, alpha=0.6, color=group_colors[group_idx])
    
#     ax_right.set_xlabel('Density', fontsize=10)
#     ax_right.tick_params(labelleft=False)
#     ax_right.grid(True, alpha=0.3)
    
#     # Set title
#     fig.suptitle(f't-SNE Visualization of Participant Embeddings by {metric_name} (Diagnosis Dataset)', 
#                 fontsize=14, fontweight='bold')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print statistics
#     print(f"\n--- {metric_name} Analysis (Diagnosis Dataset) ---")
#     print(f"{metric_name} range: {metric_values.min():.3f} - {metric_values.max():.3f}")
#     print(f"25th percentile (Q1): {q25:.3f}")
#     print(f"75th percentile (Q3): {q75:.3f}")
#     print(f"Median: {np.percentile(metric_values, 50):.3f}")
    
#     for group_idx in range(3):
#         count = np.sum(groups == group_idx)
#         avg_value = metric_values[groups == group_idx].mean()
#         print(f"{metric_name} group '{group_labels[group_idx]}': {count} participants (avg: {avg_value:.3f})")

# # Create behavioral plots for diagnosis dataset
# print("\n" + "="*60)
# print("BEHAVIORAL METRICS - DIAGNOSIS DATASET")
# print("="*60)

# # Accumulated Reward plot for diagnosis dataset
# q25_diag = np.percentile(accumulated_rewards_diag, 25)
# q75_diag = np.percentile(accumulated_rewards_diag, 75)

# reward_groups_diag = np.zeros(len(accumulated_rewards_diag), dtype=int)
# reward_groups_diag[accumulated_rewards_diag <= q25_diag] = 0
# reward_groups_diag[(accumulated_rewards_diag > q25_diag) & (accumulated_rewards_diag < q75_diag)] = 1
# reward_groups_diag[accumulated_rewards_diag >= q75_diag] = 2

# reward_group_labels_diag = [
#     f'Lower Quartile (≤{q25_diag:.1f})',
#     f'Middle 50% ({q25_diag:.1f}-{q75_diag:.1f})',
#     f'Upper Quartile (≥{q75_diag:.1f})'
# ]

# reward_colors = ['#e74c3c', '#f39c12', '#27ae60']

# fig_reward_diag = plt.figure(figsize=(12, 10))
# gs_reward_diag = fig_reward_diag.add_gridspec(3, 3, width_ratios=[1, 3, 0.3], height_ratios=[1, 3, 0.3],
#                                              hspace=0.05, wspace=0.05)

# ax_main_reward_diag = fig_reward_diag.add_subplot(gs_reward_diag[1, 1])
# ax_top_reward_diag = fig_reward_diag.add_subplot(gs_reward_diag[0, 1], sharex=ax_main_reward_diag)
# ax_right_reward_diag = fig_reward_diag.add_subplot(gs_reward_diag[1, 2], sharey=ax_main_reward_diag)

# for group_idx in range(3):
#     mask = reward_groups_diag == group_idx
#     ax_main_reward_diag.scatter(
#         embeddings_reduced_diagnosis[mask, 0], 
#         embeddings_reduced_diagnosis[mask, 1],
#         c=reward_colors[group_idx], 
#         label=reward_group_labels_diag[group_idx],
#         alpha=0.7,
#         s=50
#     )

# ax_main_reward_diag.set_xlabel('t-SNE Dimension 1', fontsize=12)
# ax_main_reward_diag.set_ylabel('t-SNE Dimension 2', fontsize=12)
# ax_main_reward_diag.legend(title='Reward Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
# ax_main_reward_diag.grid(True, alpha=0.3)

# x_range_diag = np.linspace(embeddings_reduced_diagnosis[:, 0].min(), embeddings_reduced_diagnosis[:, 0].max(), 200)
# for group_idx in range(3):
#     mask = reward_groups_diag == group_idx
#     if np.sum(mask) > 1:
#         kde = gaussian_kde(embeddings_reduced_diagnosis[mask, 0])
#         density = kde(x_range_diag)
#         ax_top_reward_diag.fill_between(x_range_diag, density, alpha=0.6, color=reward_colors[group_idx])

# ax_top_reward_diag.set_ylabel('Density', fontsize=10)
# ax_top_reward_diag.tick_params(labelbottom=False)
# ax_top_reward_diag.grid(True, alpha=0.3)

# y_range_diag = np.linspace(embeddings_reduced_diagnosis[:, 1].min(), embeddings_reduced_diagnosis[:, 1].max(), 200)
# for group_idx in range(3):
#     mask = reward_groups_diag == group_idx
#     if np.sum(mask) > 1:
#         kde = gaussian_kde(embeddings_reduced_diagnosis[mask, 1])
#         density = kde(y_range_diag)
#         ax_right_reward_diag.fill_betweenx(y_range_diag, density, alpha=0.6, color=reward_colors[group_idx])

# ax_right_reward_diag.set_xlabel('Density', fontsize=10)
# ax_right_reward_diag.tick_params(labelleft=False)
# ax_right_reward_diag.grid(True, alpha=0.3)

# fig_reward_diag.suptitle('t-SNE Visualization by Accumulated Reward (Diagnosis Dataset)', 
#                         fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.show()

# print(f"\n--- Reward Analysis (Diagnosis Dataset) ---")
# print(f"Accumulated reward range: {accumulated_rewards_diag.min():.2f} - {accumulated_rewards_diag.max():.2f}")
# print(f"25th percentile (Q1): {q25_diag:.2f}")
# print(f"75th percentile (Q3): {q75_diag:.2f}")
# print(f"Median: {np.percentile(accumulated_rewards_diag, 50):.2f}")

# for group_idx in range(3):
#     count = np.sum(reward_groups_diag == group_idx)
#     avg_reward = accumulated_rewards_diag[reward_groups_diag == group_idx].mean()
#     print(f"Reward group '{reward_group_labels_diag[group_idx]}': {count} participants (avg: {avg_reward:.2f})")

# # Switch Rate plot for diagnosis dataset
# create_behavioral_plot_diagnosis(switch_rates_diag, 'Switch Rate', embeddings_reduced_diagnosis, 'custom')

# # Stay After Reward plot for diagnosis dataset  
# create_behavioral_plot_diagnosis(stay_after_reward_rates_diag, 'Stay After Reward', embeddings_reduced_diagnosis, 'RdYlGn')

# # Perseveration plot for diagnosis dataset
# create_behavioral_plot_diagnosis(perseveration_rates_diag, 'Perseveration', embeddings_reduced_diagnosis, 'RdYlGn_r')