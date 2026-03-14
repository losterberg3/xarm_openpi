#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
    # Just inspect keys:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only

    # Convert to PyTorch:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

Example:
    # pi0_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

    # pi0_aloha_sim
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

    # pi05_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch

Custom modules (e.g. GRU / gated residual):
    If you added a module like memory_gru (GRUCell) in JAX, do two things:
    1. Use a config that has that module enabled (e.g. --config-name pi05_gru_addition for gru=True).
    2. Add the same module in PI0Pytorch when the config flag is set (see memory_gru in pi0_pytorch.py).
    The conversion script maps JAX param names to PyTorch; for GRUCell we use slice_memory_gru_state_dict.
    To add another custom module: implement a slice_*_state_dict(projection_params) and merge its result into
    projection_params in convert_pi0_checkpoint, and add the submodule in PI0Pytorch.__init__ when the config flag is set.
"""

import json
import os
import pathlib
import shutil
from typing import Literal

from flax.nnx import traversals
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma) — try expected key; fallback in convert_pi0_checkpoint adds from full params if missing
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    for jax_key in (f"llm/embedder/input_embedding{suffix}", "llm/embedder/input_embedding"):
        if jax_key in state_dict:
            state_dict[pytorch_key] = state_dict.pop(jax_key)
            break

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[
            i
        ].transpose()

        if "pi05" in checkpoint_dir:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    # Handle final norm layer
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def _param_value(p):
    """Get raw array from a param that may be wrapped as {'value': array} (NNX state)."""
    if isinstance(p, dict) and "value" in p:
        return p["value"]
    return p


def slice_memory_gru_state_dict(projection_params: dict) -> dict:
    """Convert JAX/Flax NNX GRUCell params to PyTorch nn.GRUCell state dict.

    Flax nnx.GRUCell uses dense_i (in -> 3*hidden, with bias) and dense_h (hidden -> 3*hidden, no bias).
    PyTorch nn.GRUCell expects weight_ih, weight_hh, bias_ih, bias_hh.
    """
    if "memory_gru" not in projection_params:
        return {}
    gru = projection_params["memory_gru"]
    dense_i = gru.get("dense_i", gru)
    dense_h = gru.get("dense_h", gru)
    # Handle nested 'value' wrapper from NNX state
    kernel_i = np.array(_param_value(dense_i["kernel"]))
    bias_i = np.array(_param_value(dense_i["bias"]))
    kernel_h = np.array(_param_value(dense_h["kernel"]))
    # Flax: kernel_i (in_features, 3*hidden), kernel_h (hidden, 3*hidden). PyTorch: weight_ih (3*hidden, in), weight_hh (3*hidden, hidden)
    return {
        "memory_gru.weight_ih": torch.from_numpy(kernel_i.T.copy()),
        "memory_gru.weight_hh": torch.from_numpy(kernel_h.T.copy()),
        "memory_gru.bias_ih": torch.from_numpy(bias_i.copy()),
        "memory_gru.bias_hh": torch.from_numpy(np.zeros_like(bias_i, dtype=bias_i.dtype)),
    }


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    # Use repository restore utility to load a pure dict of params (value suffix removed)
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.

    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    # Initialize checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    """
    Convert PI0 JAX checkpoint to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16)
        output_path: Path to save the converted PyTorch model
        model_config: Model config
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    # Break down orbax ckpts by restoring via JAX to respect dtype
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")

    # Process projection params (Linear layers: kernel + bias)
    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    proj_tree = initial_params["projection_params"]
    for key in keys:
        if key not in proj_tree:
            continue
        kernel_params = proj_tree[key]["kernel"]
        bias_params = proj_tree[key]["bias"]
        weight = np.array(_param_value(kernel_params))
        bias = np.array(_param_value(bias_params))

        projection_params[f"{key}.weight"] = torch.from_numpy(weight.T)
        projection_params[f"{key}.bias"] = torch.from_numpy(bias)

    # Process memory_gru (GRUCell) when config.gru is True; JAX uses dense_i / dense_h, PyTorch uses weight_ih/hh, bias_ih/hh
    if getattr(model_config, "gru", False) and "memory_gru" in proj_tree:
        gru_params = slice_memory_gru_state_dict(proj_tree)
        projection_params.update(gru_params)
        print("Converted memory_gru (GRUCell) parameters.")
    elif getattr(model_config, "gru", False) and "memory_gru" not in proj_tree:
        print("Warning: config.gru=True but checkpoint has no 'memory_gru'. Run with --inspect_only to list keys.")

    # Create configs based on checkpoint path
    # All models use the same PaliGemma config structure
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # Process PaliGemma weights
    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)

    # Ensure language embed_tokens is present (flatten_mapping can miss it in some checkpoint layouts)
    embed_tokens_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if embed_tokens_key not in paligemma_params:
        full_params = initial_params["projection_params"]
        try:
            arr = np.array(
                _param_value(full_params["PaliGemma"]["llm"]["embedder"]["input_embedding"])
            )
            paligemma_params[embed_tokens_key] = torch.from_numpy(arr)
            print("Restored embed_tokens from full params (was missing from flattened PaliGemma).")
        except KeyError as e:
            raise KeyError(
                f"Language model embed_tokens not found in checkpoint (tried flattened dict and "
                f"full params PaliGemma.llm.embedder.input_embedding). {e}"
            ) from e

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    # Instantiate model (skip transformers_replace check; we only need the structure to load weights)
    os.environ["OPENPI_SKIP_TRANSFORMERS_CHECK"] = "1"
    try:
        pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    finally:
        os.environ.pop("OPENPI_SKIP_TRANSFORMERS_CHECK", None)

    # Combine all parameters (no prefix needed for our model structure)
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # Load state dict
    pi0_model.load_state_dict(all_params, strict=False)

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    # Save the converted model using safetensors
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, "model.safetensors")

    # Save state: model state_dict plus any all_params keys that didn't load (e.g. embed_tokens).
    # save_model() only writes model.state_dict(), so keys that were "unexpected" would be missing.
    state_to_save = dict(pi0_model.state_dict())
    state_to_save.update(all_params)
    # Safetensors requires contiguous tensors; clone to avoid views/transposes.
    state_to_save = {k: v.contiguous().clone() for k, v in state_to_save.items()}
    safetensors.torch.save_file(state_to_save, out_path)

    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config as JSON for reference
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    """Load JAX model and optionally convert to PyTorch.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory
        output_path: Path to save converted PyTorch model (required for conversion)
        precision: Precision for model conversion
        inspect_only: Only inspect parameter keys, don't convert
    """
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)
