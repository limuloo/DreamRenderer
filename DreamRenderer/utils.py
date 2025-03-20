import argparse
import numpy as np
import torch
import os
import yaml
import random
from diffusers.utils.import_utils import is_accelerate_available
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import EulerDiscreteScheduler
import cv2
if is_accelerate_available():
    from accelerate import init_empty_weights
from contextlib import nullcontext


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


import torch
from typing import Callable, Dict, List, Optional, Union
from collections import defaultdict


def get_all_processor_keys(model, parent_name=''):
    all_processor_keys = []
    
    for name, module in model.named_children():
        full_name = f'{parent_name}.{name}' if parent_name else name
        
        # Check if the module has 'processor' attribute
        if hasattr(module, 'processor'):
            all_processor_keys.append(f'{full_name}.processor')
        
        # Recursively check submodules
        all_processor_keys.extend(get_all_processor_keys(module, full_name))
    
    return all_processor_keys