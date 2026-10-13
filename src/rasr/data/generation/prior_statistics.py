"""Compute the prior-statistics figure in parallel across CPU workers."""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import LogLocator, MaxNLocator
from tqdm import tqdm

from src.rasr.data.generation.data_creator import (
    _remap_expression_variables,
    _suppress_simplipy_runtime_warnings,
    load_toml_config,
)
from src.rasr.data.generation.dataset_sampler import DatasetSampler
from src.rasr.data.generation.expression_sampler import ExpressionSampler
from src.rasr.data.generation.expression_wrapper import ExpressionWrapper
from src.rasr.data.tokenization import EquationTokenizer


@dataclass
class WorkerContext:
    expression_sampler: ExpressionSampler
    dataset_sampler: DatasetSampler
    expression_wrapper: ExpressionWrapper
    tokenizer: EquationTokenizer


_WORKER_CONTEXT: WorkerContext | None = None


def _init_worker(config: dict, base_seed: int) -> None:
    global _WORKER_CONTEXT
    _suppress_simplipy_runtime_warnings()

    worker_seed = hash((base_seed, os.getpid())) & 0xFFFFFFFF
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    expression_sampler = ExpressionSampler.from_config(config)
    dataset_sampler = DatasetSampler.from_config(config)
    expression_wrapper = ExpressionWrapper(config["prior"]["expressions"]["operator_arity"])
    tok = config["tokenizer"]
    tokenizer = EquationTokenizer(
        tok["special_tokens"], tok["operators"], tok["variables"], tok["digits"], tok["special_chars"]
    )
    _WORKER_CONTEXT = WorkerContext(
        expression_sampler=expression_sampler,
        dataset_sampler=dataset_sampler,
        expression_wrapper=expression_wrapper,
        tokenizer=tokenizer,
    )


def _sample_stats(num_samples: int) -> dict | None:
    """Sample one (expression, dataset) pair and return only its summary statistics."""
    ctx = _WORKER_CONTEXT
    if ctx is None:
        raise RuntimeError("Worker not initialized correctly.")

    try:
        skeleton, variables = ctx.expression_sampler.sample_skeleton_only_with_retries(retries=10)
        x_data, y_hat, variable_mapping, noise_scale, raw_constants, mu_y, sigma_y = (
            ctx.dataset_sampler.sample_with_retries(
                skeleton=skeleton,
                variables=variables,
                num_samples=num_samples,
                retries=3,
                num_extra_columns=None,
                filter_iqr=True,
            )
        )
        expression, _ = ctx.expression_sampler.build_expression_from_skeleton(
            skeleton, raw_constants, raw_constants, variables
        )

        # Count nodes before round_constants expands each constant into digit tokens.
        unrounded_wrapped = ctx.expression_wrapper.wrap(expression, mu_y, sigma_y)
        expr_size = len(unrounded_wrapped)

        wrapped = ctx.expression_wrapper.round_constants(unrounded_wrapped, ctx.expression_sampler.sig_figs)
        remapped_wrapped = _remap_expression_variables(wrapped, variable_mapping)
        token_length = len(ctx.tokenizer.encode_tokens(remapped_wrapped)) + 1  # SOE/EOE token.

        n_unique_vars = len(variables)
        n_distractors = x_data.shape[1] - n_unique_vars

        return dict(
            n_unique_vars=n_unique_vars,
            n_distractors=n_distractors,
            noise_scale=float(noise_scale),
            n_constants=len(raw_constants),
            expr_size=expr_size,
            token_length=token_length,
            constants=[float(c) for c in raw_constants],
        )
    except Exception:
        return None


def _collect_stats(config: dict, n_stat: int, num_samples: int, num_workers: int, seed: int) -> pd.DataFrame:
    stats_rows: list[dict] = []
    all_constants: list[float] = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(config, seed),
    ) as executor:
        futures = deque()
        window_size = max(1, num_workers * 2)
        submitted = 0
        progress = tqdm(total=n_stat, desc="sampling")
        while len(stats_rows) < n_stat:
            while len(futures) < window_size and submitted < n_stat:
                futures.append(executor.submit(_sample_stats, num_samples))
                submitted += 1

            result = futures.popleft().result()
            if result is not None:
                all_constants.extend(result.pop("constants"))
                stats_rows.append(result)
                progress.update(1)

            if submitted >= n_stat and not futures and len(stats_rows) < n_stat:
                # Some samples failed; top up with a fresh batch.
                remaining = n_stat - len(stats_rows)
                for _ in range(remaining):
                    futures.append(executor.submit(_sample_stats, num_samples))
                submitted += remaining
        progress.close()

    df = pd.DataFrame(stats_rows)
    df.attrs["all_constants"] = all_constants
    return df


def _plot(df: pd.DataFrame, const_max_abs: float, out_path: Path) -> None:
    """Plot the six-panel prior-statistics histogram grid."""
    all_abs = np.abs(np.asarray(df.attrs["all_constants"]))
    all_abs = all_abs[all_abs > 0]  # Zeros are undefined on a log scale.

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )
    BAR_COLOR = "#4C72B0"

    fig, axes = plt.subplots(1, 6, figsize=(19, 3.4), constrained_layout=True)

    def _int_hist(ax, values, xlabel, bin_width=1):
        values = values.astype(int)
        lo, hi = values.min(), values.max()
        edges = np.arange(lo, hi + bin_width + 1, bin_width) - 0.5
        # No edgecolor: stroked borders on thin bars read as gaps at high sample counts.
        ax.hist(values, bins=edges, color=BAR_COLOR, edgecolor="none")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_xlabel(xlabel, fontsize=12)

    _int_hist(axes[0], df["n_unique_vars"], "Distinct variables $n_v$")
    _int_hist(axes[1], df["n_distractors"], "Distractor features")

    axes[2].hist(df["noise_scale"], bins=40, color=BAR_COLOR, edgecolor="none")
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[2].set_xlabel(r"Noise strength $s$ ($\sigma_y$ units)", fontsize=12)

    _int_hist(axes[3], df["n_constants"], "Constants per expression")

    log_bins = np.logspace(-np.log10(const_max_abs), np.log10(const_max_abs), 41)
    axes[4].hist(all_abs, bins=log_bins, color=BAR_COLOR, edgecolor="none")
    axes[4].set_xscale("log")
    axes[4].set_xlim(1.0 / const_max_abs, const_max_abs)
    axes[4].xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    axes[4].set_xlabel(r"Constant magnitude $|c|$", fontsize=12)

    # Wider bins keep bars solid over the large node-count range.
    _int_hist(axes[5], df["expr_size"], "Expression size (nodes)", bin_width=4)

    for ax in axes:
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the prior-statistics figure in parallel across CPU workers."
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to a TOML data generation config file.")
    parser.add_argument(
        "-n", "--n-stat", type=int, default=2000, help="Number of (expression, dataset) pairs to sample."
    )
    parser.add_argument("--num-samples", type=int, default=256, help="Number of rows per sampled dataset.")
    parser.add_argument("-w", "--num-workers", type=int, default=os.cpu_count(), help="Number of CPU worker processes.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prior-statistics.pdf",
        help="Output PDF path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for per-worker RNGs.")
    args = parser.parse_args()

    config = load_toml_config(args.config)

    df = _collect_stats(
        config=config,
        n_stat=args.n_stat,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Collected {len(df)} successful samples, {len(df.attrs['all_constants'])} constants")

    # The sampler rejects constants outside [1/M, M], so this is their true support.
    const_max_abs = DatasetSampler.from_config(config)._constant_max_abs

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(df, const_max_abs, out_path)


if __name__ == "__main__":
    main()
