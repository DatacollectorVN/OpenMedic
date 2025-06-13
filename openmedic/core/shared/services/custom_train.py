import torch
import torch.optim as optim
import torch.nn as nn
from typing import List, Optional, Dict
import logging

from openmedic.core.shared.services.management import ConfigReader
from openmedic.core.shared.services.objects.model import ModelBase, ModelManager
import openmedic.core.shared.services.objects.metric as metric
import openmedic.core.shared.services.objects.optimization as optimization
import openmedic.core.shared.services.objects.loss_function as lf
import openmedic.core.shared.services.objects.monitor as mt


class OpenMedicTrainerException(Exception):
    """Custom exception"""
    def __init__(self, message: str="An error occurred in OpenMedicTrainerException"):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicTrainer:
    def __init__(self,
        model: ModelBase,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
        metric_op: metric.MetricOpBase,
        monitor_map_ops: Dict[str, mt.MonitorOpBase]
    ):
        self.model: ModelBase = model
        self.optimizer: optim.Optimizer = optimizer
        self.loss_function: nn.Module = loss_function
        self.metric: metric.MetricOpBase = metric_op
        self.monitor_map_ops: Dict[str, mt.MonitorOpBase] = monitor_map_ops

    def feedforward(self, images: torch.Tensor, gts: torch.Tensor) -> list:
        gts_pred: torch.Tensor = self.model(images)
        losses: torch.Tensor = self.loss_function(gts_pred, gts)
        res: dict = self.metric.execute(gts_pred=gts_pred, gts=gts)
        metric_score = res["mean"]

        return losses, metric_score

    def get_object(self, names: List[str]) -> list:
        return [
            getattr(self, name)
            for name in names
        ]

    def _execute_check_point(self, **kwargs):
        import shared.services.objects.ops.monitors.checkpoint as checkpoint
        checkpoint_handler: Optional[checkpoint.CheckPoint] = self.monitor_map_ops.get("CheckPoint", None)
        if not checkpoint_handler:
            return

        checkpoint_handler.execute(**kwargs)

    def _monitor_kwargs_for_execution(self, **kwargs) -> dict:
        return {
            # kwargs for `CheckPoint`
            "model": self.model,
        }

    def execute_monitor(self, **kwargs):

        kwargs.update(
            self._monitor_kwargs_for_execution(**kwargs)
        )

        monitor_info: Optional[dict] = ConfigReader.get_field(name="monitor")
        if not monitor_info:
            return

        for monitor_op in self.monitor_map_ops.values():
            monitor_op.execute(**kwargs)


    @classmethod
    def initialize_with_config(cls):
        model_info: dict = ConfigReader.get_field(name="model")
        optim_info: dict = ConfigReader.get_field(name="optimization")
        loss_info: dict = ConfigReader.get_field(name="loss_function")
        metric_info: dict = ConfigReader.get_field(name="metric")
        monitor_info: Optional[dict] = ConfigReader.get_field(name="monitor")

        model: ModelBase = cls._get_model(model_info=model_info)
        optimizer: optim.Optimizer = cls._get_optimizer(model=model, optim_info=optim_info)
        loss_function: nn.modules = cls._get_loss_function(loss_info=loss_info)
        metric_op: metric.MetricOpBase = cls._get_metric(metric_info=metric_info)
        monitor_map_ops: Dict[str, mt.MonitorOpBase] = cls._get_monitor_map_ops(monitor_info=monitor_info)

        return cls(model, optimizer, loss_function, metric_op, monitor_map_ops)

    @classmethod
    def _get_model(cls, model_info: dict) -> ModelBase:
        model_name: str = model_info["name"]
        model_params: dict = model_info["params"]
        transfer_learning_model: str = model_info.get("transfer_learning_model", '')
        model: ModelBase = ModelManager.get_model(model_name=model_name)(**model_params)
        if transfer_learning_model:
            model.load_state_dict(torch.load(transfer_learning_model))

        return model

    @classmethod
    def _get_optimizer(cls, model: ModelBase, optim_info: dict) -> optim.Optimizer:
        optim_name: str = optim_info["name"]
        optim_params: str = optim_info["params"]
        params: dict = {
            "params": model.parameters()
        }
        params.update(optim_params)
        return optimization.OptimizationManager.get_torch_optimization(name=optim_name, **params)

    @classmethod
    def _get_loss_function(cls, loss_info: dict) -> nn.Module:
        loss_name: str = loss_info["name"]
        loss_params: dict = loss_info["params"]
        loss_type: str = loss_info["type"]
        if loss_type =="torch":
            return lf.LossFunctionManager.get_torch_loss_function(name=loss_name, **loss_params)
        elif loss_type=="custom":
            return lf.LossFunctionManager.get_custom_loss_function(name=loss_name, **loss_params)
        else:
            raise OpenMedicTrainerException(f"`loss_type` ({loss_type}) is not support.")

    @classmethod
    def _get_monitor_map_ops(cls, monitor_info: Optional[dict]) -> Dict[str, mt.MonitorOpBase]:
        if not monitor_info:
            logging.warning(f"[{cls.__name__}][_get_monitor_manager]: You did not configure `monitor`")
            return {}

        monitor_map_ops: Dict[str, mt.MonitorOpBase] = {}
        for op_name, params in monitor_info.items():
            if not isinstance(params, dict):
                raise OpenMedicTrainerException(f"The `{op_name}` expects its parameters as dictionay.")

            monitor_map_ops.update(
                {
                    op_name: mt.MonitorManager.get_op(op_name=op_name).initialize(**params)
                }
            )

        return monitor_map_ops

    @classmethod
    def _get_metric(cls, metric_info: dict) -> metric.MetricOpBase:
        metric_name: str = metric_info["name"]
        metric_params: dict = metric_info["params"]

        return metric.MetricManager.get_op(op_name=metric_name)(**metric_params)