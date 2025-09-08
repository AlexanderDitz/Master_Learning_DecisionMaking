import sys, os
import numpy as np
import argparse
import pickle
from typing import List, Tuple, Dict, Optional, Any
from scipy.optimize import minimize
from scipy.stats import norm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim
from resources.bandits import Agent


class gershman2018_uncertainty(torch.nn.Module):
    pass

def training():
    pass

class AgentGershman2018(Agent):
    pass

def setup_agent_gershman():
    pass

def main():
    pass

if __name__=='__main__':
    pass
