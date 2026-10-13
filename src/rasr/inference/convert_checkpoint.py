"""Convert a DCP or ``model.pt`` checkpoint directory into a ``model.safetensors`` directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from src.rasr.inference.checkpoint import WEIGHTS_FILE, load_rasr_model, save_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="Directory with DCP shards or model.pt (or a run dir) and config.toml.")
    parser.add_argument("dst", type=Path, help="New directory to write model.safetensors and config.toml to.")
    parser.add_argument("--config", type=Path, default=None, help="Defaults to <src>/config.toml.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step for run dirs; latest if omitted.")
    args = parser.parse_args()

    config_path = args.config or args.src / "config.toml"
    if args.dst.exists() and any(args.dst.iterdir()):
        raise FileExistsError(f"{args.dst} exists and is not empty.")
    args.dst.mkdir(parents=True, exist_ok=True)

    # Matches the dtype the base weights are stored in and that post-training saves.
    model, _, _ = load_rasr_model(str(args.src), str(config_path), step=args.step, dtype=torch.bfloat16)
    save_weights(model, args.dst)
    shutil.copy2(config_path, args.dst / "config.toml")
    if (args.src / "post_train.toml").is_file():
        shutil.copy2(args.src / "post_train.toml", args.dst / "post_train.toml")

    # Round-trip through the loader to make sure the new directory is usable as-is.
    state_dict = model.state_dict()
    reloaded, _, _ = load_rasr_model(str(args.dst), str(args.dst / "config.toml"), dtype=torch.bfloat16)
    for k, v in reloaded.state_dict().items():
        if not torch.equal(v, state_dict[k]):
            raise RuntimeError(f"Mismatch after reload for {k}.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Wrote {args.dst / WEIGHTS_FILE} ({len(state_dict)} tensors, {n_params:,} parameters).")


if __name__ == "__main__":
    main()
