import re
from pycocotools.coco import COCO
import io
from contextlib import redirect_stdout
from pycocotools import mask as maskUtils
import numpy as np
from typing import List
import sys
import importlib


class ModuleInterface:
    """The plugin interface"""

    @staticmethod
    def init() -> None:
        """Register the Module"""


def camel_to_snake(name: str) -> str:
    """
    Converts CamelCase or camelCase to snake_case.
    Example: 'CamelCase' -> 'camel_case'
    """
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    snake_case = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake_case


def snake_to_camel(name: str) -> str:
    components: list = name.split('_')
    return ''.join(x.capitalize() for x in components)


def load_coco_file(annotation_path: str) -> COCO:
    # Suppress stdout by COCO
    with io.StringIO() as buf, redirect_stdout(buf):
        coco: COCO = COCO(annotation_file=annotation_path)
    return coco


def convert_to_gt(ann_ids: List[dict], img_h: int, img_w) -> np.ndarray:
    gt_canvas: np.ndarray = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann_id in ann_ids:
        seg: any = ann_id['segmentation']
        category_id: int = ann_id["category_id"]
        mask: np.ndarray
        if isinstance(seg, dict):  # RLE
            if isinstance(seg['counts'], str):
                seg['counts'] = seg['counts'].encode('utf-8')
            mask = maskUtils.decode(seg)
        else:  # Polygon
            rles = maskUtils.frPyObjects(seg, img_h, img_w)
            mask = maskUtils.decode(rles)
            if len(mask.shape) == 3:
                mask = np.any(mask, axis=2)  # merge multiple polygons into one

        # Make the mask pixel value according to catgory_id
        mask[mask == 1] = category_id
        gt_canvas = np.maximum(gt_canvas, mask.astype(np.uint8))

    return gt_canvas


def import_module(module_name: str) -> any:
    if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
        # https://bobbyhadz.com/blog/python-importerror-cannot-import-name-mapping-from-collections
        import collections.abc
        collections.Mapping = collections.abc.Mapping
        collections.MutableMapping = collections.abc.MutableMapping
        return importlib.import_module(module_name)


class BreakLoop(Exception):
    """Custom exception"""
    def __init__(self, message: str=''):
        self.message: str = message
        super().__init__(self.message)