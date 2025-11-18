from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.utils import has_offloaded_params
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core.session_functions import LifecycleCallbacks
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import dispatch_for_sequential
from llmcompressor.utils.dev import dispatch_for_generation

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["DataFreePipeline"]


@CalibrationPipeline.register("datafree")
class DataFreePipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        dataset_args: "DatasetArguments",
    ):
        """
        A pipeline for data-free calibration

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        # Check if model already has offloaded params (sequential dispatch)
        # If so, keep the sequential dispatch setup
        has_sequential_dispatch = any(
            has_offloaded_params(module) for module in model.modules()
        )
        
        if not has_sequential_dispatch:
            # Standard behavior: dispatch for generation
            dispatch_for_generation(model)

        LifecycleCallbacks.calibration_epoch_start()
        LifecycleCallbacks.calibration_epoch_end()
