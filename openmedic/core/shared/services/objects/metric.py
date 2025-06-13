import torch
from abc import ABC, abstractmethod

import openmedic.core.shared.services.utils as utils


class MetricOpBase(ABC):
    @classmethod
    def get_name(cls):
        return cls.__name__
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def execute(self) -> dict:
        pass


class MetricOpError(Exception):
    """Custom exception"""
    def __init__(self, message: str=''):
        self.message: str = message
        super().__init__(self.message)


class MetricManager:
    @classmethod
    def add_op(cls, op_name: str, op_class: MetricOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> MetricOpBase:
        error_msg: str = ''
        try:
            return getattr(cls, op_name)
        except AttributeError:
            error_msg=f"The {op_name} operation does not exist."
            raise MetricOpError(message=error_msg)
        

