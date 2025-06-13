from abc import ABC, abstractmethod
from typing import List

import openmedic.core.shared.services.utils as utils
import openmedic.core.shared.services.objects.model as model
import openmedic.core.shared.services.objects.transform as transform
import openmedic.core.shared.services.objects.metric as metric
import openmedic.core.shared.services.objects.loss_function as lf
import openmedic.core.shared.services.objects.monitor as monitor


class BaseRegister(ABC):
    __TEMPLATE: str
    __MODULES: List[str]

    @abstractmethod
    def import_modules(cls):
       pass

    @abstractmethod
    def register():
        pass


class ModelRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.model_zoo.{module}"
    __MODULES: List[str] = [
        "unet"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, model_class: model.ModelBase, model_name: str=''):
        if not model_name:
            model_name = model_class.get_name()
        model.ModelManager.add_model(model_name=model_name, model_class=model_class)


class TransformRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.transformers.{module}"
    __MODULES: List[str] = [
        "resize",
        "flip"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            model_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            model_interface.init()

    @classmethod
    def register(cls, transform_class: transform.TransformOpBase, transfrom_name: str=''):
        if not transfrom_name:
            transfrom_name = transform_class.get_name()
        transform.TransformManager.add_op(op_name=transfrom_name, op_class=transform_class)


class MetricRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.metrics.{module}"
    __MODULES: List[str] = [
        "dice_score"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, metric_class: metric.MetricOpBase, metric_name: str=''):
        if not metric_name:
            metric_name = metric_class.get_name()
        metric.MetricManager.add_op(op_name=metric_name, op_class=metric_class)


class LossRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.losses.{module}"
    __MODULES: List[str] = [
        "dice_loss"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, loss_class: lf.LossOpBase, loss_name: str=''):
        if not loss_name:
            loss_name = loss_class.get_name()
        lf.LossFunctionManager.add_op(op_name=loss_name, op_class=loss_class)


class MonitorRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.monitors.{module}"
    __MODULES: List[str] = [
        "checkpoint"
    ]
    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, monitor_class: lf.LossOpBase, monitor_name: str=''):
        if not monitor_name:
            monitor_name = monitor_class.get_name()
        monitor.MonitorManager.add_op(op_name=monitor_name, op_class=monitor_class)


class Regsiter:
    model_register: ModelRegister = ModelRegister
    transform_register: TransformRegister = TransformRegister
    metric_register: MetricRegister = MetricRegister
    loss_register: LossRegister = LossRegister
    monitor_register: MonitorRegister = MonitorRegister

    @classmethod
    def init(cls):
        cls.model_register.import_modules()
        cls.transform_register.import_modules()
        cls.metric_register.import_modules()
        cls.loss_register.import_modules()
        cls.monitor_register.import_modules()