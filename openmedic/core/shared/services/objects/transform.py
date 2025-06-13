from typing import List
from abc import ABC, abstractmethod
import numpy as np


class TransformOpBase(ABC):
    @classmethod
    def get_name(cls):
        return cls.__name__

    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def initialize():
        pass

    @abstractmethod
    def execute(image: np.ndarray, gt: np.ndarray) -> List[np.ndarray]:
        pass


class TransformOpError(Exception):
    """Custom exception"""
    def __init__(self, message: str=''):
        self.message: str = message
        super().__init__(self.message)


class TransformManager:
    def __call__() -> List[object]:
        pass

    @classmethod
    def add_op(cls, op_name: str, op_class: TransformOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> TransformOpBase:
        error_msg: str = ''
        try:
            return getattr(cls, op_name)
        except AttributeError:
            error_msg=f"The {op_name} operation does not exist."
            raise TransformOpError(message=error_msg)