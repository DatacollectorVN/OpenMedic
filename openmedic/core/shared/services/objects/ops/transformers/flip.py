from dataclasses import dataclass
from typing import List
import numpy as np

from openmedic.core.shared.services.objects.transform import TransformOpBase
import openmedic.core.shared.services.objects.registry as registry


@dataclass
class Flip(TransformOpBase):
   def __init__(self):
      pass
   
   @classmethod
   def initialize(cls, *args, **kwargs):
      return cls()
   
   def execute(self, image: np.ndarray, gt: np.ndarray) -> List[np.ndarray]:
      return image, gt
   
def init():
   registry.TransformRegister.register(transform_class=Flip)