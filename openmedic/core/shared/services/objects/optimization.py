import torch.optim as optim
import torch.nn as nn

import openmedic.core.shared.services.utils as utils


class OptimizationOpBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return utils.camel_to_snake(cls.__name__)
    

class OptimizationOpError(Exception):
    """Custom exception"""
    def __init__(self, message: str=''):
        self.message: str = message
        super().__init__(self.message)


class OptimizationManager:
    @staticmethod
    def get_torch_optimization(name: str, **kwargs) -> optim.Optimizer:
        return getattr(optim, name)(**kwargs)