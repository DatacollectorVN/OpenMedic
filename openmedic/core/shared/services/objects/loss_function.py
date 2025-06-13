import torch.nn as nn
import torch

from abc import ABC, abstractmethod

from openmedic.core.shared.services.management import ConfigReader
import openmedic.core.shared.services.utils as utils


class LossOpBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.__name__


class LossOpError(Exception):
    """Custom exception"""
    def __init__(self, message: str=''):
        self.message: str = message
        super().__init__(self.message)


class LossFunctionManager:
    @classmethod
    def _preprocess_torch(cls, **kwargs) -> dict:
        # Process `weight` parameter from list to torch.Tensor
        if kwargs.get("weight", None):
            device: str = "cpu"
            if ConfigReader.get_field(name="pipeline").get("is_gpu", False):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kwargs["weight"] = torch.tensor(kwargs["weight"], device=device)
        
        return kwargs

    @classmethod
    def get_torch_loss_function(cls, name: str, **kwargs) -> nn.Module:
        kwargs: dict = cls._preprocess_torch(**kwargs)
        return getattr(nn, name)(**kwargs)
    
    @classmethod
    def get_custom_loss_function(cls, name: str, **kwargs) -> nn.Module:        
        return getattr(cls, name)(**kwargs)

    @classmethod
    def add_op(cls, op_name: str, op_class: LossOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> LossOpBase:
        error_msg: str = ''
        try:
            return getattr(cls, op_name)
        except AttributeError:
            error_msg=f"The {op_name} operation does not exist."
            raise LossOpError(message=error_msg)
