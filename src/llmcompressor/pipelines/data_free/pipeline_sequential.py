"""
Sequential data-free pipeline for quantization without calibration data.
This enables sequential offloading for data-free quantization like FP8.
"""
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core.session_functions import LifecycleCallbacks
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import dispatch_for_sequential

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialDataFreePipeline"]


@CalibrationPipeline.register("sequential_datafree")
class SequentialDataFreePipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        dataset_args: "DatasetArguments",
    ):
        """
        A pipeline for data-free calibration with sequential offloading support.
        
        Unlike the standard DataFreePipeline which calls dispatch_for_generation
        (loading the entire model to GPU), this uses dispatch_for_sequential to
        enable layer-by-layer processing for large models.

        :param model: model being calibrated
        :param dataloader: loads data for calibration (unused for data-free)
        :param dataset_args: dataset arguments relevant to pipelines
        """
        # Set up sequential offloading instead of loading entire model to GPU
        dispatch_for_sequential(model)

        LifecycleCallbacks.calibration_epoch_start()
        LifecycleCallbacks.calibration_epoch_end()