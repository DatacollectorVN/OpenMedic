import torch
from typing import Union

import openmedic.core.shared.services.objects.metric as metric
import openmedic.core.shared.services.objects.registry as registry


class DiceCoefficient(metric.MetricOpBase):
    def __init__(self, n_classes: int, include_background: bool, epsilon: Union[float, str]=1e-5, is_detail: bool=True):
        """
        Input:
        ------
            n_classes: total number of classes (including background)
            include_background: whether to include class 0 in Dice calculation
            epsilon: small value to avoid divide-by-zero
        """
        self.n_classes: int = n_classes
        self.include_background: bool = include_background
        self.epsilon: float = float(epsilon)
        self.is_detail: bool = is_detail

    def execute(self, gts_pred: torch.Tensor, gts: torch.Tensor) -> dict:
        """
        Computes Dice coefficient per class for multi-class segmentation.

        Input:
        ------
            gts_pred: model logits, shape [B, C, H, W]
            gts: ground truth class indices, shape [B, H, W]

        Output:
        -------
            mean_dice: average Dice across present classes
        """
        pred_classes: torch.Tensor = torch.argmax(gts_pred, dim=1)  # [B, H, W]
        dice_scores: list = []

        class_range: range = range(self.n_classes) if self.include_background else range(1, self.n_classes)

        for cls in class_range:
            pred_cls: torch.Tensor = (pred_classes == cls).float()  # [B, H, W]
            label_cls: torch.Tensor = (gts == cls).float()          # [B, H, W]

            # Skip this class if not present in the batch
            if label_cls.sum() == 0:
                continue

            intersection: torch.Tensor = (pred_cls * label_cls).sum(dim=(1, 2))  # [B]
            union: torch.Tensor = pred_cls.sum(dim=(1, 2)) + label_cls.sum(dim=(1, 2))  # [B]

            dice: torch.Tensor = (2 * intersection + self.epsilon) / (union + self.epsilon)  # [B]
            dice_scores.append(dice.mean())  # scalar

        if len(dice_scores) > 0:
            mean_dice = torch.stack(dice_scores).mean().item()
        else:
            mean_dice = 0.0  # No classes present

        if self.is_detail:
            score_per_class: list = [d.item() for d in dice_scores]

        return {
            "mean": mean_dice,
            # "score_per_class": [d.item() for d in dice_scores]
        }


def init():
    registry.MetricRegister.register(metric_class=DiceCoefficient)