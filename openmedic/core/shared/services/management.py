from typing import Dict
import yaml
import argparse
from abc import ABC, abstractmethod

import openmedic.core.shared.services.utils as utils


class ConfigException(Exception):
    """Custom exception"""
    def __init__(self, message: str="An error occurred in ConfigReader"):
        self.message: str = message
        super().__init__(self.message)


class ConfigReader:
    @classmethod
    def _init(cls, sections: Dict[str, dict]):
        for section,  content in sections.items():
            setattr(cls, section, content)

    @classmethod
    def initialize(cls, config_path: str):
        if not config_path.endswith(".yml") and \
            not config_path.endswith(".yaml"):
            raise ConfigException("Only support `yaml` or `yml` file.")

        with open(config_path, 'r') as stream:
            cls._init(sections=yaml.safe_load(stream))


    @classmethod
    def _check_required_field(cls, name: str, attr_fields: list):
        required_fields: list = []
        if name == "data":
            required_fields = ["image_dir", "coco_annotation_path"]
        elif name == "model":
            required_fields = ["name", "params"]
        elif name in ["optimization", "metric"]:
            required_fields = ["name", "params"]
        elif name == "loss_function":
            required_fields = ["name", "params", "type"]
        elif name == "pipeline":
            required_fields = ["batch_size", "n_epochs", "train_ratio"]

        is_subset: bool = set(required_fields).issubset(set(attr_fields))
        if not is_subset:
            error_msg = f"The required fields in `{name}`: {required_fields}"
            raise ConfigException(message=error_msg)

    @classmethod
    def get_field(cls, name: str) -> any:
        error_msg: str = ''
        try:
            attr: any =  getattr(cls, name)
            if name in ["transform", "data", "model", "optimization", "loss_function", "pipeline", "metric", "monitor"]:
                if not isinstance(attr, dict):
                    error_msg = f"The field `{name}` need to be parsed as dictionary."
                    raise ConfigException(message=error_msg)

                attr_fields: list = list(attr.keys())
                cls._check_required_field(name=name, attr_fields=attr_fields)
            return attr

        except AttributeError:
            if name in ["transform", "monitor"]:
                # Return None if config file does not include `transform` or `monitor` field
                return None
            error_msg = f"The field `{name}` does not exist in config file."
            raise ConfigException(message=error_msg)


class OpenMedicPipelineBase(ABC):
    @abstractmethod
    def init_arguments():
        pass

    @abstractmethod
    def run() -> dict:
        pass


class OpenMedicPipeline:
    MODULE_TEMPLATE: str = "pipelines.{pipeline_name}"

    @classmethod
    def execute(cls, pipeline_name: str) -> dict:
        args: argparse.Namespace
        parser: argparse.ArgumentParser
        pipeline: OpenMedicPipelineBase = utils.import_module(
            module_name=cls.MODULE_TEMPLATE.format(
                pipeline_name=pipeline_name,
            )
        )
        parser = pipeline.init_arguments()
        args, _ = parser.parse_known_args()
        kwargs = {
            attr: getattr(args, attr) for attr in dir(args) if not attr.startswith("_")
        }
        kwargs["pipeline_name"] = pipeline_name
        return pipeline.run(**kwargs)


