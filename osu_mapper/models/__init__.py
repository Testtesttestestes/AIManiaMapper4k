"""
Neural network models for osu!mania 4K AI mapper pipeline.
Four distinct AI architectures:
1. SyncBPMOffsetModel (Model 1: BPM and Offset audio synchronization)
2. DynamicsSVModel (Model 2: Slider Velocity and timing dynamics)
3. ManiaPatternTransformer (Model 3: Causal autoregressive pattern generator)
4. DifficultyEvaluatorModel (Model 4: Star Rating and density validation)
"""

from .sync_model import SyncBPMOffsetModel, snap_offset_to_audio_transients
from .dynamics_model import DynamicsSVModel
from .transformer import (
    ManiaPatternTransformer,
    TransformerConfig,
    calculate_pattern_loss,
    keys_to_token,
    token_to_keys,
)
from .evaluator_model import DifficultyEvaluatorModel

__all__ = [
    "SyncBPMOffsetModel",
    "snap_offset_to_audio_transients",
    "DynamicsSVModel",
    "ManiaPatternTransformer",
    "TransformerConfig",
    "DifficultyEvaluatorModel",
    "calculate_pattern_loss",
    "keys_to_token",
    "token_to_keys",
]

