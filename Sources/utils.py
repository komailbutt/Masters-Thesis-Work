# Sources/utils.py

import random
import numpy as np
import torch

def set_seed(seed=42):
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)

   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

   print(f"Seed set to {seed}")

def get_device():
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("Using device:", device)
   return device
 
