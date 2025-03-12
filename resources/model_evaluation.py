import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Union
import numpyro

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import get_update_dynamics
from benchmarking.hierarchical_bayes_numpyro import rl_model


def log_likelihood(data: np.ndarray, probs: np.ndarray, axis: int = None, normalization: int = 1):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1 
    
    # Sum over all data points and negate the result
    return np.sum(np.sum(data * np.log(probs), axis=-1), axis=axis) / normalization


def bayesian_information_criterion(data: np.ndarray, probs: np.ndarray, n_parameters: int, ll: np.ndarray = None, axis: int = None, normalization: int = 1):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if ll is None:
        ll = log_likelihood(daa=data, probs=probs, axis=axis, normalization=normalization)
    
    return -2 * ll + n_parameters * np.log(np.prod(data.shape[:-1]))

def akaike_information_criterion(data: np.ndarray, probs: np.ndarray, n_parameters: int, ll: np.ndarray = None, axis: int = None, normalization: int = 1):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if ll is None:
        ll = log_likelihood(data=data, probs=probs, axis=axis, normalization=normalization)
    
    return -2 * ll + 2 * n_parameters

def get_scores(data: np.ndarray, probs: np.ndarray, n_parameters: int, axis: int = None, normalization: int = 1) -> float:
        ll = log_likelihood(data=data, probs=probs, axis=axis, normalization=normalization)
        bic = bayesian_information_criterion(data=data, probs=probs, n_parameters=n_parameters, ll=ll, axis=axis, normalization=normalization)
        aic = akaike_information_criterion(data=data, probs=probs, n_parameters=n_parameters, ll=ll, axis=axis, normalization=normalization)
        return -ll, bic, aic
    
def plot_traces(file_numpyro: Union[str, numpyro.infer.MCMC], figsize=(12, 8)):
    """
    Plot trace plots for posterior samples.

    Parameters:
    - samples: dict, where keys are parameter names and values are arrays of samples.
    - param_names: list of str, parameter names to include in the plot.
    - figsize: tuple, size of the figure.
    """
    plt.rc('font', size=7)
    plt.rc('axes', titlesize=7)
    plt.rc('axes', labelsize=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    
    if isinstance(file_numpyro, str):
        with open(file_numpyro, 'rb') as file:
            mcmc = pickle.load(file)
    elif isinstance(file_numpyro, numpyro.infer.MCMC):
        mcmc = file_numpyro
    else:
        raise AttributeError('argument 0 (file_numpyro) is not of class str or numpyro.infer.MCMC.')
    
    samples = mcmc.get_samples()
    param_names = list(mcmc.get_samples().keys())
    
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 2]})

    for i, param in enumerate(param_names):
        param_samples = samples[param]
        
        # Trace plot
        axes[i, 1].plot(param_samples, alpha=0.7, linewidth=0.7)
        # axes[i, 1].set_title(f"Trace Plot: {param}")
        # axes[i, 1].set_ylabel(param)
        # axes[i, 1].set_xlabel("Iteration")

        # KDE plot
        sns.kdeplot(param_samples, ax=axes[i, 0], fill=True, color="skyblue", legend=False)
        # axes[i, 0].set_title(f"Posterior: {param}")
        # axes[i, 0].set_xlabel(param)
        y_label = param
        check_for = ['mean', 'std']
        for check in check_for:
            if check in y_label:
                # remove param name and keep only remaining part (mean or std)
                param = param[param.find(check):]
        axes[i, 0].set_ylabel(param)

    plt.tight_layout()
    plt.show()