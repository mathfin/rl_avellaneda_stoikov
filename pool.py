import numpy as np
import torch
import random

from config import SEED
from ppo_model.ppo_manager import PPO


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


ppo_agent = PPO(state_dim=5,
                action_dim=2,
                lr_actor=0.0003,
                lr_critic=0.001,
                gamma=0.99,
                K_epochs=40,
                eps_clip=0.1)