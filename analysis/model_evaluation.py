"""
Model Evaluation Metrics:

This script computes quantitative metrics for analyzing reward sensitivity and clinical differences
in model dynamics. Implements vector magnitude analysis, directional consistency, and reward 
contrast ratios for comparing LSTM, SPICE, RNN, and GQL models.

Metrics implemented:
1. Vector Magnitude Analysis - How much models' states change after rewards vs no rewards
2. Directional Consistency - Whether reward-driven state changes follow consistent patterns  
3. Reward Contrast Ratio - Relative sensitivity to positive vs negative outcomes
4. State Space Exploration - How broadly models explore their internal state space
5. Temporal Dynamics Analysis - How reward history influences future state updates
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import pickle
import torch
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model comparison functions
from model_comparison import (
    load_dezfouli_dataset, 
    load_lstm_model, load_spice_model, load_rnn_model, load_gql_model,
    extract_neural_dynamics_dezfouli, extract_spice_dynamics_dezfouli, extract_gql_dynamics_dezfouli
)


class ModelEvaluationMetrics:
    """
    Class for computing and analyzing model dynamics metrics.
    """
    
    def __init__(self, models_loaded: Dict[str, Any], dataset: Dict[str, pd.DataFrame]):
        """
        Initialize the metrics evaluator.
        
        Args:
            models_loaded: Dictionary of loaded models
            dataset: Dezfouli dataset with participant data
        """
        self.models_loaded = models_loaded
        self.dataset = dataset
        self.metrics_results = {}
        
    def compute_vector_magnitude_analysis(self, 
                                        participant_ids: Optional[List[str]] = None,
                                        reward_conditions: List[int] = [0, 1]) -> Dict[str, Dict]:
        """
        Compute vector magnitude analysis for reward sensitivity by diagnosis group and model.
        
        Args:
            participant_ids: List of participant IDs to analyze (if None, uses all)
            reward_conditions: List of reward conditions [0=unrewarded, 1=rewarded]
            
        Returns:
            Dictionary with magnitude statistics per model, diagnosis, and condition
        """
        print("\n📊 Computing Vector Magnitude Analysis by Diagnosis Group...")
        
        # Group participants by diagnosis
        diagnosis_groups = {'Healthy': [], 'Depression': [], 'Bipolar': []}
        for participant_id, participant_data in self.dataset.items():
            if 'diagnosis' in participant_data.columns:
                diagnosis = participant_data['diagnosis'].iloc[0]
                if diagnosis in diagnosis_groups:
                    diagnosis_groups[diagnosis].append(participant_id)
        
        print(f"  Participants by diagnosis:")
        for diagnosis, participants in diagnosis_groups.items():
            print(f"    {diagnosis}: {len(participants)} participants")
        
        magnitude_results = {}
        
        for model_name in ['LSTM', 'SPICE', 'RNN', 'GQL']:
            if self.models_loaded.get(model_name) is None:
                continue
                
            print(f"\n  Analyzing {model_name} model...")
            model = self.models_loaded[model_name]
            
            magnitude_results[model_name] = {}
            
            # Analyze each diagnosis group separately
            for diagnosis, participant_ids in diagnosis_groups.items():
                if len(participant_ids) == 0:
                    continue
                
                print(f"    Processing {diagnosis} group...")
                
                magnitude_results[model_name][diagnosis] = {
                    'rewarded': [],
                    'unrewarded': [],
                    'all_magnitudes': [],
                    'participant_data': {}
                }
                
                for participant_id in participant_ids[:10]:  # Limit to first 10 for efficiency
                    if participant_id not in self.dataset:
                        continue
                        
                    participant_data = self.dataset[participant_id]
                    actions = participant_data['choice'].values
                    rewards_left = participant_data['reward_left'].values
                    rewards_right = participant_data['reward_right'].values
                    rewards = np.column_stack([rewards_left, rewards_right])
                    
                    # Extract dynamics
                    if model_name in ['LSTM', 'RNN']:
                        states, state_changes = extract_neural_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'SPICE':
                        states, state_changes = extract_spice_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'GQL':
                        states, state_changes = extract_gql_dynamics_dezfouli(model, rewards, actions)
                    
                    if len(state_changes) == 0:
                        continue
                    
                    # Compute vector magnitudes
                    magnitudes = np.linalg.norm(state_changes, axis=1)
                    
                    # Get trial rewards for each state change
                    trial_rewards = rewards[np.arange(len(actions)-1), actions[:-1]]  # -1 because state_changes is one shorter
                    
                    # Separate by reward condition
                    rewarded_mask = (trial_rewards == 1)
                    unrewarded_mask = (trial_rewards == 0)
                    
                    rewarded_magnitudes = magnitudes[rewarded_mask]
                    unrewarded_magnitudes = magnitudes[unrewarded_mask]
                    
                    magnitude_results[model_name][diagnosis]['rewarded'].extend(rewarded_magnitudes)
                    magnitude_results[model_name][diagnosis]['unrewarded'].extend(unrewarded_magnitudes)
                    magnitude_results[model_name][diagnosis]['all_magnitudes'].extend(magnitudes)
                    
                    # Store per-participant data
                    magnitude_results[model_name][diagnosis]['participant_data'][participant_id] = {
                        'rewarded_mean': np.mean(rewarded_magnitudes) if len(rewarded_magnitudes) > 0 else 0,
                        'unrewarded_mean': np.mean(unrewarded_magnitudes) if len(unrewarded_magnitudes) > 0 else 0,
                        'rewarded_std': np.std(rewarded_magnitudes) if len(rewarded_magnitudes) > 0 else 0,
                        'unrewarded_std': np.std(unrewarded_magnitudes) if len(unrewarded_magnitudes) > 0 else 0,
                        'n_rewarded': len(rewarded_magnitudes),
                        'n_unrewarded': len(unrewarded_magnitudes)
                    }
                
                # Compute summary statistics for this diagnosis group
                if len(magnitude_results[model_name][diagnosis]['rewarded']) > 0 and len(magnitude_results[model_name][diagnosis]['unrewarded']) > 0:
                    magnitude_results[model_name][diagnosis]['summary'] = {
                        'rewarded_mean': np.mean(magnitude_results[model_name][diagnosis]['rewarded']),
                        'unrewarded_mean': np.mean(magnitude_results[model_name][diagnosis]['unrewarded']),
                        'rewarded_std': np.std(magnitude_results[model_name][diagnosis]['rewarded']),
                        'unrewarded_std': np.std(magnitude_results[model_name][diagnosis]['unrewarded']),
                        'reward_contrast_ratio': (
                            np.mean(magnitude_results[model_name][diagnosis]['rewarded']) / 
                            np.mean(magnitude_results[model_name][diagnosis]['unrewarded'])
                            if np.mean(magnitude_results[model_name][diagnosis]['unrewarded']) > 0 else np.inf
                        ),
                        'effect_size': self._compute_cohens_d(
                            magnitude_results[model_name][diagnosis]['rewarded'],
                            magnitude_results[model_name][diagnosis]['unrewarded']
                        )
                    }
                    
                    print(f"      ✓ {diagnosis}: Rewarded mean={magnitude_results[model_name][diagnosis]['summary']['rewarded_mean']:.4f}, "
                          f"Unrewarded mean={magnitude_results[model_name][diagnosis]['summary']['unrewarded_mean']:.4f}, "
                          f"Contrast ratio={magnitude_results[model_name][diagnosis]['summary']['reward_contrast_ratio']:.3f}")
                else:
                    magnitude_results[model_name][diagnosis]['summary'] = {
                        'rewarded_mean': 0, 'unrewarded_mean': 0, 'rewarded_std': 0, 'unrewarded_std': 0,
                        'reward_contrast_ratio': 0, 'effect_size': 0
                    }
        
        self.metrics_results['vector_magnitude'] = magnitude_results
        return magnitude_results
    
    def compute_directional_consistency(self, 
                                      participant_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Compute directional consistency of state changes within reward conditions by diagnosis group.
        
        Args:
            participant_ids: List of participant IDs to analyze
            
        Returns:
            Dictionary with directional consistency metrics per model and diagnosis
        """
        print("\n🧭 Computing Directional Consistency Analysis by Diagnosis Group...")
        
        # Group participants by diagnosis
        diagnosis_groups = {'Healthy': [], 'Depression': [], 'Bipolar': []}
        for participant_id, participant_data in self.dataset.items():
            if 'diagnosis' in participant_data.columns:
                diagnosis = participant_data['diagnosis'].iloc[0]
                if diagnosis in diagnosis_groups:
                    diagnosis_groups[diagnosis].append(participant_id)
        
        consistency_results = {}
        
        for model_name in ['LSTM', 'SPICE', 'RNN', 'GQL']:
            if self.models_loaded.get(model_name) is None:
                continue
                
            print(f"\n  Analyzing {model_name} model...")
            model = self.models_loaded[model_name]
            
            consistency_results[model_name] = {}
            
            # Analyze each diagnosis group separately
            for diagnosis, participant_ids in diagnosis_groups.items():
                if len(participant_ids) == 0:
                    continue
                
                print(f"    Processing {diagnosis} group...")
                
                consistency_results[model_name][diagnosis] = {
                    'rewarded_vectors': [],
                    'unrewarded_vectors': [],
                    'rewarded_consistency': 0,
                    'unrewarded_consistency': 0,
                    'participant_data': {}
                }
                
                for participant_id in participant_ids[:10]:  # Limit to first 10 for efficiency
                    if participant_id not in self.dataset:
                        continue
                        
                    participant_data = self.dataset[participant_id]
                    actions = participant_data['choice'].values
                    rewards_left = participant_data['reward_left'].values
                    rewards_right = participant_data['reward_right'].values
                    rewards = np.column_stack([rewards_left, rewards_right])
                    
                    # Extract dynamics
                    if model_name in ['LSTM', 'RNN']:
                        states, state_changes = extract_neural_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'SPICE':
                        states, state_changes = extract_spice_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'GQL':
                        states, state_changes = extract_gql_dynamics_dezfouli(model, rewards, actions)
                    
                    if len(state_changes) == 0:
                        continue
                    
                    # Normalize state changes to unit vectors
                    normalized_changes = state_changes / (np.linalg.norm(state_changes, axis=1, keepdims=True) + 1e-8)
                    
                    # Get trial rewards
                    trial_rewards = rewards[np.arange(len(actions)-1), actions[:-1]]
                    
                    # Separate by reward condition
                    rewarded_mask = (trial_rewards == 1)
                    unrewarded_mask = (trial_rewards == 0)
                    
                    rewarded_vectors = normalized_changes[rewarded_mask]
                    unrewarded_vectors = normalized_changes[unrewarded_mask]
                    
                    consistency_results[model_name][diagnosis]['rewarded_vectors'].extend(rewarded_vectors)
                    consistency_results[model_name][diagnosis]['unrewarded_vectors'].extend(unrewarded_vectors)
                
                # Compute consistency as mean pairwise cosine similarity
                if len(consistency_results[model_name][diagnosis]['rewarded_vectors']) > 1:
                    rewarded_vectors = np.array(consistency_results[model_name][diagnosis]['rewarded_vectors'])
                    rewarded_similarity = np.mean([
                        np.dot(rewarded_vectors[i], rewarded_vectors[j])
                        for i in range(len(rewarded_vectors))
                        for j in range(i+1, len(rewarded_vectors))
                    ])
                    consistency_results[model_name][diagnosis]['rewarded_consistency'] = rewarded_similarity
                
                if len(consistency_results[model_name][diagnosis]['unrewarded_vectors']) > 1:
                    unrewarded_vectors = np.array(consistency_results[model_name][diagnosis]['unrewarded_vectors'])
                    unrewarded_similarity = np.mean([
                        np.dot(unrewarded_vectors[i], unrewarded_vectors[j])
                        for i in range(len(unrewarded_vectors))
                        for j in range(i+1, len(unrewarded_vectors))
                    ])
                    consistency_results[model_name][diagnosis]['unrewarded_consistency'] = unrewarded_similarity
                
                print(f"      ✓ {diagnosis}: Rewarded consistency={consistency_results[model_name][diagnosis]['rewarded_consistency']:.4f}, "
                      f"Unrewarded consistency={consistency_results[model_name][diagnosis]['unrewarded_consistency']:.4f}")
        
        self.metrics_results['directional_consistency'] = consistency_results
        return consistency_results
    
    def compute_reward_contrast_ratio_by_diagnosis(self, 
                                                 diagnosis_groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, Dict]:
        """
        Compute reward contrast ratios separated by clinical diagnosis.
        
        Args:
            diagnosis_groups: Dictionary mapping diagnosis to participant IDs
            
        Returns:
            Dictionary with contrast ratios per model and diagnosis
        """
        print("\n🎯 Computing Reward Contrast Ratios by Diagnosis...")
        
        if diagnosis_groups is None:
            # Group participants by diagnosis
            diagnosis_groups = {'Healthy': [], 'Depression': [], 'Bipolar': []}
            for participant_id, participant_data in self.dataset.items():
                if 'diagnosis' in participant_data.columns:
                    diagnosis = participant_data['diagnosis'].iloc[0]
                    if diagnosis in diagnosis_groups:
                        diagnosis_groups[diagnosis].append(participant_id)
        
        contrast_results = {}
        
        for model_name in ['LSTM', 'SPICE', 'RNN', 'GQL']:
            if self.models_loaded.get(model_name) is None:
                continue
                
            print(f"  Analyzing {model_name} model...")
            model = self.models_loaded[model_name]
            
            contrast_results[model_name] = {}
            
            for diagnosis, participant_ids in diagnosis_groups.items():
                if len(participant_ids) == 0:
                    continue
                
                print(f"    Processing {diagnosis} group ({len(participant_ids)} participants)...")
                
                all_rewarded_magnitudes = []
                all_unrewarded_magnitudes = []
                
                for participant_id in participant_ids[:5]:  # Limit for efficiency
                    if participant_id not in self.dataset:
                        continue
                        
                    participant_data = self.dataset[participant_id]
                    actions = participant_data['choice'].values
                    rewards_left = participant_data['reward_left'].values
                    rewards_right = participant_data['reward_right'].values
                    rewards = np.column_stack([rewards_left, rewards_right])
                    
                    # Extract dynamics
                    if model_name in ['LSTM', 'RNN']:
                        states, state_changes = extract_neural_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'SPICE':
                        states, state_changes = extract_spice_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'GQL':
                        states, state_changes = extract_gql_dynamics_dezfouli(model, rewards, actions)
                    
                    if len(state_changes) == 0:
                        continue
                    
                    # Compute magnitudes and separate by reward
                    magnitudes = np.linalg.norm(state_changes, axis=1)
                    trial_rewards = rewards[np.arange(len(actions)-1), actions[:-1]]
                    
                    rewarded_magnitudes = magnitudes[trial_rewards == 1]
                    unrewarded_magnitudes = magnitudes[trial_rewards == 0]
                    
                    all_rewarded_magnitudes.extend(rewarded_magnitudes)
                    all_unrewarded_magnitudes.extend(unrewarded_magnitudes)
                
                # Compute contrast ratio for this diagnosis group
                if len(all_rewarded_magnitudes) > 0 and len(all_unrewarded_magnitudes) > 0:
                    contrast_ratio = np.mean(all_rewarded_magnitudes) / np.mean(all_unrewarded_magnitudes)
                    effect_size = self._compute_cohens_d(all_rewarded_magnitudes, all_unrewarded_magnitudes)
                    
                    contrast_results[model_name][diagnosis] = {
                        'contrast_ratio': contrast_ratio,
                        'rewarded_mean': np.mean(all_rewarded_magnitudes),
                        'unrewarded_mean': np.mean(all_unrewarded_magnitudes),
                        'rewarded_std': np.std(all_rewarded_magnitudes),
                        'unrewarded_std': np.std(all_unrewarded_magnitudes),
                        'effect_size': effect_size,
                        'n_rewarded': len(all_rewarded_magnitudes),
                        'n_unrewarded': len(all_unrewarded_magnitudes)
                    }
                    
                    print(f"      ✓ {diagnosis}: Contrast ratio={contrast_ratio:.3f}, Effect size={effect_size:.3f}")
        
        self.metrics_results['reward_contrast_by_diagnosis'] = contrast_results
        return contrast_results
    
    def compute_state_space_exploration(self, 
                                      participant_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Compute state space exploration metrics by diagnosis group.
        
        Args:
            participant_ids: List of participant IDs to analyze
            
        Returns:
            Dictionary with exploration metrics per model and diagnosis
        """
        print("\n🗺️ Computing State Space Exploration Analysis by Diagnosis Group...")
        
        # Group participants by diagnosis
        diagnosis_groups = {'Healthy': [], 'Depression': [], 'Bipolar': []}
        for participant_id, participant_data in self.dataset.items():
            if 'diagnosis' in participant_data.columns:
                diagnosis = participant_data['diagnosis'].iloc[0]
                if diagnosis in diagnosis_groups:
                    diagnosis_groups[diagnosis].append(participant_id)
        
        exploration_results = {}
        
        for model_name in ['LSTM', 'SPICE', 'RNN', 'GQL']:
            if self.models_loaded.get(model_name) is None:
                continue
                
            print(f"\n  Analyzing {model_name} model...")
            model = self.models_loaded[model_name]
            
            exploration_results[model_name] = {}
            
            # Analyze each diagnosis group separately
            for diagnosis, participant_ids in diagnosis_groups.items():
                if len(participant_ids) == 0:
                    continue
                
                print(f"    Processing {diagnosis} group...")
                
                all_states_rewarded = []
                all_states_unrewarded = []
                
                for participant_id in participant_ids[:10]:  # Limit to first 10 for efficiency
                    if participant_id not in self.dataset:
                        continue
                        
                    participant_data = self.dataset[participant_id]
                    actions = participant_data['choice'].values
                    rewards_left = participant_data['reward_left'].values
                    rewards_right = participant_data['reward_right'].values
                    rewards = np.column_stack([rewards_left, rewards_right])
                    
                    # Extract dynamics
                    if model_name in ['LSTM', 'RNN']:
                        states, state_changes = extract_neural_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'SPICE':
                        states, state_changes = extract_spice_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'GQL':
                        states, state_changes = extract_gql_dynamics_dezfouli(model, rewards, actions)
                    
                    if len(states) == 0:
                        continue
                    
                    # Get trial rewards
                    trial_rewards = rewards[np.arange(len(actions)-1), actions[:-1]]
                    
                    # Separate states by reward condition
                    rewarded_states = states[trial_rewards == 1]
                    unrewarded_states = states[trial_rewards == 0]
                    
                    all_states_rewarded.extend(rewarded_states)
                    all_states_unrewarded.extend(unrewarded_states)
                
                # Compute exploration metrics
                if len(all_states_rewarded) > 0:
                    states_rewarded = np.array(all_states_rewarded)
                    rewarded_hull_area = self._compute_convex_hull_area(states_rewarded)
                    rewarded_std = np.mean(np.std(states_rewarded, axis=0))
                else:
                    rewarded_hull_area = 0
                    rewarded_std = 0
                
                if len(all_states_unrewarded) > 0:
                    states_unrewarded = np.array(all_states_unrewarded)
                    unrewarded_hull_area = self._compute_convex_hull_area(states_unrewarded)
                    unrewarded_std = np.mean(np.std(states_unrewarded, axis=0))
                else:
                    unrewarded_hull_area = 0
                    unrewarded_std = 0
                
                exploration_results[model_name][diagnosis] = {
                    'rewarded_hull_area': rewarded_hull_area,
                    'unrewarded_hull_area': unrewarded_hull_area,
                    'rewarded_std': rewarded_std,
                    'unrewarded_std': unrewarded_std,
                    'exploration_ratio': rewarded_hull_area / unrewarded_hull_area if unrewarded_hull_area > 0 else np.inf
                }
                
                print(f"      ✓ {diagnosis}: Rewarded area={rewarded_hull_area:.4f}, "
                      f"Unrewarded area={unrewarded_hull_area:.4f}, "
                      f"Exploration ratio={exploration_results[model_name][diagnosis]['exploration_ratio']:.3f}")
        
        self.metrics_results['state_space_exploration'] = exploration_results
        return exploration_results
    
    def compute_temporal_dynamics(self, 
                                participant_ids: Optional[List[str]] = None,
                                window_size: int = 5) -> Dict[str, Dict]:
        """
        Compute temporal dynamics analysis by diagnosis group (how reward history influences future updates).
        
        Args:
            participant_ids: List of participant IDs to analyze
            window_size: Size of temporal window for analysis
            
        Returns:
            Dictionary with temporal dynamics metrics per model and diagnosis
        """
        print("\n⏰ Computing Temporal Dynamics Analysis by Diagnosis Group...")
        
        # Group participants by diagnosis
        diagnosis_groups = {'Healthy': [], 'Depression': [], 'Bipolar': []}
        for participant_id, participant_data in self.dataset.items():
            if 'diagnosis' in participant_data.columns:
                diagnosis = participant_data['diagnosis'].iloc[0]
                if diagnosis in diagnosis_groups:
                    diagnosis_groups[diagnosis].append(participant_id)
        
        temporal_results = {}
        
        for model_name in ['LSTM', 'SPICE', 'RNN', 'GQL']:
            if self.models_loaded.get(model_name) is None:
                continue
                
            print(f"\n  Analyzing {model_name} model...")
            model = self.models_loaded[model_name]
            
            temporal_results[model_name] = {}
            
            # Analyze each diagnosis group separately
            for diagnosis, participant_ids in diagnosis_groups.items():
                if len(participant_ids) == 0:
                    continue
                
                print(f"    Processing {diagnosis} group...")
                
                autocorrelations = []
                reward_decay_effects = []
                
                for participant_id in participant_ids[:10]:  # Limit to first 10 for efficiency
                    if participant_id not in self.dataset:
                        continue
                        
                    participant_data = self.dataset[participant_id]
                    actions = participant_data['choice'].values
                    rewards_left = participant_data['reward_left'].values
                    rewards_right = participant_data['reward_right'].values
                    rewards = np.column_stack([rewards_left, rewards_right])
                    
                    # Extract dynamics
                    if model_name in ['LSTM', 'RNN']:
                        states, state_changes = extract_neural_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'SPICE':
                        states, state_changes = extract_spice_dynamics_dezfouli(model, rewards, actions)
                    elif model_name == 'GQL':
                        states, state_changes = extract_gql_dynamics_dezfouli(model, rewards, actions)
                    
                    if len(state_changes) < window_size * 2:
                        continue
                    
                    # Compute autocorrelation of state changes
                    magnitudes = np.linalg.norm(state_changes, axis=1)
                    if len(magnitudes) > window_size:
                        autocorr = self._compute_autocorrelation(magnitudes, max_lag=window_size)
                        autocorrelations.append(autocorr)
                    
                    # Compute reward decay effects
                    trial_rewards = rewards[np.arange(len(actions)-1), actions[:-1]]
                    reward_effects = self._compute_reward_decay_effect(magnitudes, trial_rewards, window_size)
                    if len(reward_effects) > 0:
                        reward_decay_effects.extend(reward_effects)
                
                # Aggregate results for this diagnosis group
                temporal_results[model_name][diagnosis] = {
                    'mean_autocorrelation': np.mean(autocorrelations, axis=0) if autocorrelations else np.zeros(window_size),
                    'autocorr_decay_rate': self._compute_decay_rate(np.mean(autocorrelations, axis=0)) if autocorrelations else 0,
                    'reward_decay_slope': np.mean(reward_decay_effects) if reward_decay_effects else 0,
                    'temporal_stability': np.std(autocorrelations) if autocorrelations else 0
                }
                
                print(f"      ✓ {diagnosis}: Decay rate={temporal_results[model_name][diagnosis]['autocorr_decay_rate']:.4f}, "
                      f"Reward decay slope={temporal_results[model_name][diagnosis]['reward_decay_slope']:.4f}")
        
        self.metrics_results['temporal_dynamics'] = temporal_results
        return temporal_results
    
    def generate_comprehensive_report(self, save_path: str = "model_evaluation_report.html") -> str:
        """
        Generate a comprehensive HTML report with all metrics and visualizations.
        
        Args:
            save_path: Path to save the HTML report
            
        Returns:
            Path to the generated report
        """
        print("\n📄 Generating Comprehensive Evaluation Report...")
        
        # Create visualizations
        self._create_comparison_plots()
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Report saved to: {save_path}")
        return save_path
    
    def export_metrics_to_csv(self, save_path: str = "model_metrics.csv") -> str:
        """
        Export all computed metrics to a CSV file for further analysis.
        
        Args:
            save_path: Path to save the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        print("\n💾 Exporting Metrics to CSV...")
        
        # Flatten all metrics into a DataFrame
        rows = []
        
        for metric_type, metric_data in self.metrics_results.items():
            if metric_type == 'vector_magnitude':
                for model_name, model_data in metric_data.items():
                    for diagnosis, data in model_data.items():
                        if 'summary' in data:
                            row = {
                                'metric_type': metric_type,
                                'model': model_name,
                                'diagnosis': diagnosis,
                                'condition': 'summary',
                                **data['summary']
                            }
                            rows.append(row)
            
            elif metric_type == 'directional_consistency':
                for model_name, model_data in metric_data.items():
                    for diagnosis, data in model_data.items():
                        row = {
                            'metric_type': metric_type,
                            'model': model_name,
                            'diagnosis': diagnosis,
                            'rewarded_consistency': data.get('rewarded_consistency', 0),
                            'unrewarded_consistency': data.get('unrewarded_consistency', 0)
                        }
                        rows.append(row)
            
            elif metric_type == 'state_space_exploration':
                for model_name, model_data in metric_data.items():
                    for diagnosis, data in model_data.items():
                        row = {
                            'metric_type': metric_type,
                            'model': model_name,
                            'diagnosis': diagnosis,
                            **data
                        }
                        rows.append(row)
            
            elif metric_type == 'temporal_dynamics':
                for model_name, model_data in metric_data.items():
                    for diagnosis, data in model_data.items():
                        row = {
                            'metric_type': metric_type,
                            'model': model_name,
                            'diagnosis': diagnosis,
                            'autocorr_decay_rate': data.get('autocorr_decay_rate', 0),
                            'reward_decay_slope': data.get('reward_decay_slope', 0),
                            'temporal_stability': data.get('temporal_stability', 0)
                        }
                        rows.append(row)
            
            elif metric_type == 'reward_contrast_by_diagnosis':
                for model_name, diagnosis_data in metric_data.items():
                    for diagnosis, data in diagnosis_data.items():
                        row = {
                            'metric_type': metric_type,
                            'model': model_name,
                            'diagnosis': diagnosis,
                            **data
                        }
                        rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        
        print(f"✅ Metrics exported to: {save_path}")
        return save_path
    
    # Helper methods
    
    def _compute_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        if len(group1) == 0 or len(group2) == 0:
            return 0
        
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _compute_convex_hull_area(self, points: np.ndarray) -> float:
        """Compute convex hull area for 2D points."""
        if len(points) < 3 or points.shape[1] < 2:
            return 0
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points[:, :2])  # Use first 2 dimensions
            return hull.volume  # In 2D, volume is area
        except:
            # Fallback: use bounding box area
            return (np.max(points[:, 0]) - np.min(points[:, 0])) * (np.max(points[:, 1]) - np.min(points[:, 1]))
    
    def _compute_autocorrelation(self, signal: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation function."""
        autocorr = []
        signal = signal - np.mean(signal)
        
        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(1.0)
            else:
                if len(signal) > lag:
                    corr = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr.append(0)
        
        return np.array(autocorr)
    
    def _compute_reward_decay_effect(self, magnitudes: np.ndarray, rewards: np.ndarray, window_size: int) -> List[float]:
        """Compute how reward effects decay over time."""
        decay_effects = []
        
        for i in range(len(rewards) - window_size):
            if rewards[i] == 1:  # After a reward
                future_magnitudes = magnitudes[i+1:i+1+window_size]
                if len(future_magnitudes) == window_size:
                    # Fit linear decay
                    x = np.arange(window_size)
                    slope, _, _, _, _ = stats.linregress(x, future_magnitudes)
                    decay_effects.append(slope)
        
        return decay_effects
    
    def _compute_decay_rate(self, autocorr: np.ndarray) -> float:
        """Compute exponential decay rate from autocorrelation."""
        if len(autocorr) < 2:
            return 0
        
        # Fit exponential decay: y = exp(-λx)
        x = np.arange(len(autocorr))
        y = autocorr
        
        # Use log transform and linear regression
        try:
            log_y = np.log(np.maximum(y, 1e-10))  # Avoid log(0)
            slope, _, _, _, _ = stats.linregress(x, log_y)
            return -slope  # Return positive decay rate
        except:
            return 0
    
    def _create_comparison_plots(self):
        """Create comparison plots for all metrics."""
        # This would create matplotlib plots for the metrics
        # Implementation would depend on specific visualization needs
        pass
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        # This would generate a comprehensive HTML report
        # Implementation would create formatted HTML with metrics tables and plots
        return "<html><body><h1>Model Evaluation Report</h1><p>Report content would go here...</p></body></html>"


def main():
    """
    Main function to run comprehensive model evaluation.
    """
    print("=== Comprehensive Model Evaluation ===")
    print()
    
    # Load models and dataset (reuse from model_comparison.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    params_dir = os.path.join(project_dir, "params", "dezfouli2019")
    
    # Model paths
    lstm_path = os.path.join(params_dir, "lstm_dezfouli2019.pkl")
    spice_path = os.path.join(params_dir, "spice_dezfouli2019_l2_0_001.pkl")
    rnn_path = os.path.join(params_dir, "rnn_dezfouli2019_l2_0_001.pkl")
    gql_path = os.path.join(params_dir, "gql_dezfouli2019_PhiChiBetaKappaC.pkl")
    
    # Load models
    models_loaded = {}
    
    try:
        models_loaded['LSTM'] = load_lstm_model(lstm_path)
        print("✓ LSTM model loaded")
    except:
        models_loaded['LSTM'] = None
        print("✗ Failed to load LSTM model")
    
    try:
        models_loaded['SPICE'] = load_spice_model(spice_path, rnn_path)
        print("✓ SPICE model loaded")
    except:
        models_loaded['SPICE'] = None
        print("✗ Failed to load SPICE model")
    
    try:
        models_loaded['RNN'] = load_rnn_model(rnn_path)
        print("✓ RNN model loaded")
    except:
        models_loaded['RNN'] = None
        print("✗ Failed to load RNN model")
    
    try:
        models_loaded['GQL'] = load_gql_model(gql_path)
        print("✓ GQL model loaded")
    except:
        models_loaded['GQL'] = None
        print("✗ Failed to load GQL model")
    
    # Load dataset
    dataset = load_dezfouli_dataset()
    if dataset is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluationMetrics(models_loaded, dataset)
    
    # Compute all metrics
    print("\n🔬 Computing All Evaluation Metrics...")
    
    # 1. Vector Magnitude Analysis
    evaluator.compute_vector_magnitude_analysis()
    
    # 2. Directional Consistency
    evaluator.compute_directional_consistency()
    
    # 3. Reward Contrast Ratio by Diagnosis
    evaluator.compute_reward_contrast_ratio_by_diagnosis()
    
    # 4. State Space Exploration
    evaluator.compute_state_space_exploration()
    
    # 5. Temporal Dynamics
    evaluator.compute_temporal_dynamics()
    
    # Export results
    evaluator.export_metrics_to_csv("model_evaluation_metrics.csv")
    evaluator.generate_comprehensive_report("model_evaluation_report.html")
    
    print("\n✅ Model evaluation complete!")
    print("📊 Results saved to:")
    print("  - model_evaluation_metrics.csv")
    print("  - model_evaluation_report.html")


if __name__ == "__main__":
    main()
