from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen4_exp.qwen4_exp import (
    Model,
    norm_weights_are_one_centered,
)


def _model_for_sanitize():
    model = Model.__new__(Model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            tie_word_embeddings=False, num_hidden_layers=0, ple_storage=False
        )
    )
    return model


def _norms(center, n=8):
    return {
        f"language_model.model.layers.{i}.self_attn.q_norm.weight": mx.full(
            (4,), center
        )
        for i in range(n)
    }


def test_mtp_weights_are_dropped_under_their_checkpoint_names():
    weights = _norms(0.0)
    weights["language_model.mtp.layers.0.self_attn.q_proj.weight"] = mx.zeros((4, 4))
    weights["mtp.layers.0.self_attn.k_proj.weight"] = mx.zeros((4, 4))

    sanitized = _model_for_sanitize().sanitize(weights)

    assert not any("mtp." in key for key in sanitized)


def test_one_centered_norms_are_shifted_to_zero_centered():
    sanitized = _model_for_sanitize().sanitize(_norms(1.0))

    key = "language_model.model.layers.0.self_attn.q_norm.weight"
    assert sanitized[key].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_zero_centered_norms_are_left_alone():
    sanitized = _model_for_sanitize().sanitize(_norms(0.0))

    key = "language_model.model.layers.0.self_attn.q_norm.weight"
    assert sanitized[key].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_shifting_is_idempotent():
    once = _model_for_sanitize().sanitize(_norms(1.0))
    twice = _model_for_sanitize().sanitize(once)

    key = "language_model.model.layers.0.self_attn.q_norm.weight"
    assert twice[key].tolist() == once[key].tolist()


def test_detection_ignores_the_vision_tower():
    vision = {
        f"vision_tower.blocks.{i}.self_attn.q_norm.weight": mx.full((4,), 1.0)
        for i in range(8)
    }

    assert norm_weights_are_one_centered(vision) is False


def test_detection_survives_a_high_magnitude_outlier():
    weights = _norms(0.0)
    weights["language_model.model.hyper_connection_mixer.hc_norm.weight"] = mx.full(
        (4,), 3.75
    )

    assert norm_weights_are_one_centered(weights) is False


def test_gated_norms_are_not_shifted():
    weights = _norms(1.0)
    gated = "language_model.model.layers.0.linear_attn.norm.weight"
    weights[gated] = mx.full((4,), 1.0)

    sanitized = _model_for_sanitize().sanitize(weights)

    assert sanitized[gated].tolist() == [1.0, 1.0, 1.0, 1.0]
