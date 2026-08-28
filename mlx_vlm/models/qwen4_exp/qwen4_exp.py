import re

import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import sanitize_key
from ..switch_layers import (
    move_expert_projection,
    split_expert_projection,
    stack_expert_projection,
)
from .config import ModelConfig
from .fp8 import convert_qwen4_exp_fp8_weights
from .language import LanguageModel
from .vision import VisionModel

_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)(?=\.)")

# Norms fed by ``Qwen4ExpRMSNorm``, which applies ``y * (1.0 + weight)`` and so
# expects zero-centered weights. ``Qwen4ExpRMSNormGated`` is deliberately absent:
# it initializes to ones and genuinely wants one-centered weights.
NORM_WEIGHT_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_conv.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".indexer.q_layernorm.weight",
    ".indexer.k_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


def _is_shiftable_norm(key, value):
    return (
        value.ndim == 1
        and key.endswith(NORM_WEIGHT_SUFFIXES)
        and "visual" not in key
        and not key.startswith("vision_tower.")
    )


def norm_weights_are_one_centered(weights):
    """Whether this checkpoint stores Qwen4ExpRMSNorm weights centered at one.

    Decided by measuring the weights rather than by inferring from checkpoint
    provenance. ``Qwen4ExpRMSNorm`` applies ``y * (1.0 + weight)``, so weights
    written for it sit near zero; conversions that fold the ``+1`` in at
    conversion time sit near one. Measuring means a checkpoint in either
    convention loads correctly, and re-running sanitize on already-shifted
    weights is a no-op rather than a corruption.

    The median is deliberate: ``hyper_connection_mixer.hc_norm`` is a genuine
    high-magnitude outlier and would drag a mean.
    """
    samples = sorted(
        float(mx.mean(value.astype(mx.float32)))
        for key, value in weights.items()
        if _is_shiftable_norm(key, value)
    )
    if not samples:
        return False
    return samples[len(samples) // 2] > 0.5


class Model(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # Measured on the raw checkpoint, before any key is dropped or renamed.
        shift_norm_weights = norm_weights_are_one_centered(weights)

        # The MTP predictor is a separate speculative artifact and is not part
        # of the base conditional-generation model. Checkpoints name these
        # ``language_model.mtp.*``, so a ``mtp.``-prefix test misses all 76 of
        # them and ``load_weights(strict=True)`` rejects the load.
        weights = {
            key: value
            for key, value in weights.items()
            if ".mtp." not in key and not key.startswith("mtp.")
        }
        weights = convert_qwen4_exp_fp8_weights(weights)
        if self.config.text_config.ple_storage:
            weights = {
                key: value
                for key, value in weights.items()
                if ".ple.ple_embedding.ngram_embedding." not in key
            }

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{layer_idx}.mlp"
            for name in ("gate_proj", "up_proj", "down_proj"):
                stack_expert_projection(
                    weights,
                    f"{prefix}.experts",
                    name,
                    f"{prefix}.switch_mlp.{name}",
                    self.config.text_config.num_experts,
                )
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights or f"{gate_up_key}.weight" in weights:
                for name in ("gate_up_proj", "down_proj"):
                    key = f"{prefix}.experts.{name}"
                    if f"{key}_scales" in weights:
                        weights[f"{key}.scales"] = weights.pop(f"{key}_scales")
                split_expert_projection(
                    weights,
                    gate_up_key,
                    [f"{prefix}.switch_mlp.gate_proj", f"{prefix}.switch_mlp.up_proj"],
                )
                move_expert_projection(
                    weights,
                    f"{prefix}.experts.down_proj",
                    f"{prefix}.switch_mlp.down_proj",
                )

        # Mirrors qwen3_5, with the sign reversed. nn.RMSNorm there applies
        # ``x * weight`` and wants one-centered weights, so qwen3_5 adds 1.0;
        # Qwen4ExpRMSNorm here applies ``y * (1.0 + weight)`` and wants
        # zero-centered weights, so a one-centered checkpoint must have 1.0
        # subtracted. Left unshifted it computes ~1.9x instead of ~0.9x on every
        # affected norm, which loads cleanly under strict=True, logs nothing,
        # and generates fluent-looking nonsense.
        sanitized = {}
        for key, value in weights.items():
            key = sanitize_key(key)
            key = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if shift_norm_weights and _is_shiftable_norm(key, value):
                value = value - 1.0
            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
