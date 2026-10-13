"""Internal helpers for loading a RASRTransformer from a torchtitan run directory."""

from __future__ import annotations

import dataclasses
import os
import re
import sys
import tempfile

import torch
from safetensors.torch import load_model, save_model
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.config.manager import ConfigManager

from src.rasr.data.tokenization import EquationTokenizer
from src.rasr.model.args import RASRModelArgs
from src.rasr.model.model import RASRTransformer

WEIGHTS_FILE = "model.safetensors"


def _load_job_config(config_path: str):
    """Load a torchtitan JobConfig purely from a TOML file, without CLI overrides."""
    _original_parse_args = ConfigManager.parse_args

    def _patched_parse_args(self, args: list[str] = sys.argv[1:]):
        toml_values = self._maybe_load_toml(args)
        config_cls = self._maybe_add_custom_config(args, toml_values)
        base_config = (
            self._dict_to_dataclass(config_cls, toml_values)
            if toml_values
            else config_cls()
        )
        # Point at the run directory so torchtitan doesn't warn about its default path.
        base_config.model.hf_assets_path = os.path.dirname(os.path.abspath(config_path))
        self.config = base_config
        self._validate_config()
        return self.config

    ConfigManager.parse_args = _patched_parse_args
    try:
        manager = ConfigManager()
        config = manager.parse_args([f"--job.config_file={config_path}"])
    finally:
        ConfigManager.parse_args = _original_parse_args

    return config


def _fill_tokenizer_ids(job_config) -> None:
    """Populate runtime token IDs on ``job_config.tokenizer``, normally done by the dataloader builder."""
    tokenizer = EquationTokenizer.from_config(job_config.tokenizer)
    vocab = tokenizer.get_vocab()

    tok_cfg = job_config.tokenizer
    if tok_cfg.trg_vocab_size <= 0:
        tok_cfg.trg_vocab_size = len(tokenizer)
    if tok_cfg.soe_token_id < 0:
        tok_cfg.soe_token_id = vocab.get("<SOE>", -1)
    if tok_cfg.eoe_token_id < 0:
        tok_cfg.eoe_token_id = vocab.get("<EOE>", -1)
    if tok_cfg.pad_token_id < 0:
        tok_cfg.pad_token_id = vocab.get("<PAD>", -1)


def _find_latest_step(checkpoint_folder: str) -> str:
    """Return the path to the highest-numbered ``step-N`` subdirectory."""
    pattern = re.compile(r"step-(\d+)$")
    best_step = -1
    best_path = ""
    for entry in os.listdir(checkpoint_folder):
        m = pattern.match(entry)
        if not m:
            continue
        step_dir = os.path.join(checkpoint_folder, entry)
        if not os.path.isdir(step_dir):
            continue
        step_num = int(m.group(1))
        if step_num > best_step:
            best_step = step_num
            best_path = step_dir
    if not best_path:
        raise FileNotFoundError(f"No step-N subdirectory found in {checkpoint_folder}")
    return best_path


def _extract_model_state(state_dict: object) -> dict:
    """Return the model-only state dict from various checkpoint formats."""
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            return state_dict["model"]
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint did not contain a state dict")
    return state_dict


def _filter_to_model_keys(state_dict: dict, model: "RASRTransformer") -> dict:
    """Drop non-model entries when a training checkpoint is loaded."""
    model_keys = set(model.state_dict().keys())
    if not model_keys:
        return state_dict
    extras = [k for k in state_dict.keys() if k not in model_keys]
    if not extras:
        return state_dict
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    if not filtered:
        raise ValueError("Checkpoint did not contain model weights")
    return filtered


def _strip_compiled_keys(state_dict: dict) -> dict:
    """Strip torch.compile infix so keys align with inference model."""
    stripped: dict[str, object] = {}
    for k, v in state_dict.items():
        stripped[k.replace("._orig_mod.", ".")] = v
    return stripped


def _load_dcp_into_model(checkpoint_dir: str, model: "RASRTransformer") -> None:
    """Load a DCP checkpoint into ``model`` via a temporary torch save, so no process group is needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "model.pt")
        dcp_to_torch_save(checkpoint_dir, tmp_path)
        state_dict = torch.load(tmp_path, map_location="cpu", weights_only=True)

    state_dict = _extract_model_state(state_dict)
    stripped = _strip_compiled_keys(state_dict)
    stripped = _filter_to_model_keys(stripped, model)
    model.load_state_dict(stripped)


def save_weights(model: "RASRTransformer", directory: str | os.PathLike) -> None:
    """Write ``model``'s weights to ``<directory>/model.safetensors``."""
    save_model(model, os.path.join(directory, WEIGHTS_FILE))


def _load_flat_pt_into_model(flat_model_path: str, model: "RASRTransformer") -> None:
    """Load a plain ``model.pt`` state dict into ``model`` in-place."""
    state_dict = torch.load(flat_model_path, map_location="cpu", weights_only=True)
    state_dict = _extract_model_state(state_dict)
    stripped = _strip_compiled_keys(state_dict)
    stripped = _filter_to_model_keys(stripped, model)
    model.load_state_dict(stripped)


def load_rasr_model(
    model_path: str,
    config_path: str,
    *,
    step: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[RASRTransformer, EquationTokenizer, object]:
    """Load ``(model, tokenizer, job_config)`` from a run directory, using the latest step if ``step`` is None."""
    job_config = _load_job_config(config_path)
    _fill_tokenizer_ids(job_config)

    # Start from the flavor preset so architecture hyperparameters match training.
    flavor = job_config.model.flavor
    try:
        from src.rasr import rasr_configs
        base_args = rasr_configs[flavor]
    except (ImportError, KeyError):
        base_args = RASRModelArgs()

    model_args = dataclasses.replace(base_args)  # Copy to avoid mutating the global preset.
    model_args.update_from_config(job_config)

    model = RASRTransformer(model_args)

    # Also accept a directory holding the weights directly (e.g. a post-training "final" dir)
    # instead of the standard <model_path>/<checkpoint.folder>/step-N/ layout.
    direct_safetensors = os.path.join(model_path, WEIGHTS_FILE)
    direct_model = os.path.join(model_path, "model.pt")
    direct_dcp_metadata = os.path.join(model_path, ".metadata")
    if step is None and os.path.isfile(direct_safetensors):
        load_model(model, direct_safetensors)
    elif step is None and os.path.isfile(direct_model):
        _load_flat_pt_into_model(direct_model, model)
    elif step is None and os.path.exists(direct_dcp_metadata):
        _load_dcp_into_model(model_path, model)
    else:
        checkpoint_folder = os.path.join(model_path, job_config.checkpoint.folder)
        if step is not None:
            step_dir = os.path.join(checkpoint_folder, f"step-{step}")
            if not os.path.isdir(step_dir):
                raise FileNotFoundError(f"Checkpoint step directory not found: {step_dir}")
        else:
            step_dir = _find_latest_step(checkpoint_folder)

        dcp_metadata = os.path.join(step_dir, ".metadata")
        flat_safetensors = os.path.join(step_dir, WEIGHTS_FILE)
        flat_model = os.path.join(step_dir, "model.pt")

        if os.path.isfile(flat_safetensors):
            load_model(model, flat_safetensors)
        elif os.path.exists(dcp_metadata):
            _load_dcp_into_model(step_dir, model)
        elif os.path.isfile(flat_model):
            _load_flat_pt_into_model(flat_model, model)
        else:
            try:
                _load_dcp_into_model(step_dir, model)
            except Exception:
                if os.path.isfile(flat_model):
                    _load_flat_pt_into_model(flat_model, model)
                else:
                    raise

    model = model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = EquationTokenizer.from_config(job_config.tokenizer)
    return model, tokenizer, job_config
