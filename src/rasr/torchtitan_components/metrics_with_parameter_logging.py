"""Torchtitan metrics processor with parameter statistics logging."""

import io
import json
import logging
import math
import os
from datetime import datetime
from typing import Any

import toml as toml_lib
import tomli_w
from torchtitan.components.metrics import MetricsProcessor, build_metrics_processor
from torchtitan.config import JobConfig as BaseJobConfig
from torchtitan.distributed import ParallelDims

from src.rasr.job_config import JobConfig
from src.rasr.torchtitan_components.parameter_logger import ParameterStatsLogger


class EnhancedMetricsProcessor(MetricsProcessor):
    """MetricsProcessor extended with parameter statistics logging."""

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: str | None = None,
    ):
        # Must be set before super().__init__, which may trigger the property setters below.
        self.param_logger = None
        self._model_parts = None

        param_logging_enabled = hasattr(job_config, "parameter_logging") and job_config.parameter_logging.enabled

        if param_logging_enabled:
            self.param_logger = ParameterStatsLogger(
                config=job_config.parameter_logging,
                model_parts=None,
                optimizers=None,
            )

        super().__init__(job_config, parallel_dims, tag)

        # Redirect TensorBoard output into the experiment folder.
        if job_config.metrics.enable_tensorboard and hasattr(self.logger, '_loggers'):
            from datetime import datetime

            from torchtitan.components.metrics import TensorBoardLogger
            from torchtitan.tools.logging import logger

            experiment_dir = self._resolve_experiment_dir(job_config)
            tb_dir = os.path.join(
                experiment_dir,
                job_config.metrics.save_tb_folder,
                datetime.now().strftime("%Y%m%d-%H%M")
            )

            for i, log in enumerate(self.logger._loggers):
                if isinstance(log, TensorBoardLogger):
                    self.logger._loggers[i] = TensorBoardLogger(tb_dir, tag)
                    logger.info(f"TensorBoard logging redirected to experiment folder: {tb_dir}")
                    break

        from torchtitan.tools.logging import logger
        if param_logging_enabled:
            logger.info(f"Parameter logging enabled with interval: {job_config.parameter_logging.log_interval}")
        else:
            logger.info("Parameter logging disabled")

        self._save_config_to_toml(job_config)

    @staticmethod
    def _resolve_experiment_dir(job_config: "JobConfig") -> str:
        """Return the experiment output directory (dump_folder)."""
        return job_config.job.dump_folder

    @staticmethod
    def _filter_none_recursive(obj: Any) -> Any:
        """Recursively drop None values from dicts, since TOML has no null."""
        if isinstance(obj, dict):
            return {k: EnhancedMetricsProcessor._filter_none_recursive(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [EnhancedMetricsProcessor._filter_none_recursive(item) for item in obj]
        else:
            return obj

    def _save_config_to_toml(self, job_config: JobConfig) -> None:
        """Save job config to TOML file in dump_folder (rank 0 only)."""
        import torch.distributed as dist
        from torchtitan.tools.logging import logger

        if not dist.is_initialized() or dist.get_rank() != 0:
            return

        if tomli_w is None and toml_lib is None:
            logger.warning("No TOML writer available, config saving disabled")
            return

        dump_folder = job_config.job.dump_folder
        os.makedirs(dump_folder, exist_ok=True)
        config_path = os.path.join(dump_folder, "config.toml")

        try:
            config_dict = self._filter_none_recursive(job_config.to_dict())
            if tomli_w is not None:
                with open(config_path, "wb") as f:
                    tomli_w.dump(config_dict, f)
            else:
                with open(config_path, "w") as f:
                    toml_lib.dump(config_dict, f)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise

    def _ensure_training_log_file(self, job_config: JobConfig) -> None:
        """Add a rank-0 training log file handler in dump_folder."""
        import torch.distributed as dist
        from torchtitan.tools.logging import logger

        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                return

            dump_folder = job_config.job.dump_folder
            if not dump_folder:
                return

            os.makedirs(dump_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(dump_folder, f"training_{timestamp}.log")

            for handler in logger.handlers:
                if getattr(handler, "baseFilename", None) == log_path:
                    return

            handler = logging.FileHandler(log_path)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.info(f"Training log file enabled at {log_path}")
        except Exception as e:
            logger.warning(f"Failed to set up training log file: {e}")

    def _collect_writers(self) -> list:
        """Return all active TensorBoard SummaryWriter instances."""
        writers = []
        logger_obj = getattr(self, "logger", None)
        if logger_obj is None:
            return writers
        writer = getattr(logger_obj, "writer", None)
        if writer is not None:
            writers.append(writer)
        loggers = getattr(logger_obj, "_loggers", None)
        if loggers:
            for sub in loggers:
                sub_writer = getattr(sub, "writer", None)
                if sub_writer is not None:
                    writers.append(sub_writer)
        return writers

    @staticmethod
    def _build_param_table(model_parts) -> str:
        """Build a Flax-style parameter table with module hierarchy and shapes.

        Identical sibling modules are collapsed into the first one, annotated with [×N].
        """
        from collections import OrderedDict, defaultdict

        all_params: OrderedDict[str, tuple] = OrderedDict()
        for part in model_parts:
            for ppath, p in part.named_parameters():
                if ppath not in all_params:
                    all_params[ppath] = (tuple(p.shape), p.numel(), p.requires_grad)

        grand_total = sum(v[1] for v in all_params.values())
        if grand_total == 0:
            return "Model parameter breakdown: (no parameters found)"

        def _module_total(prefix: str) -> int:
            if prefix == "":
                return grand_total
            pfx = prefix + "."
            return sum(v[1] for k, v in all_params.items() if k == prefix or k.startswith(pfx))

        def _sig(prefix: str) -> frozenset:
            pfx = prefix + "." if prefix else ""
            items = []
            for k, v in all_params.items():
                if prefix == "" or k == prefix or k.startswith(pfx):
                    rel = k[len(pfx):] if k.startswith(pfx) else (k if k != prefix else "")
                    items.append((rel, v[0]))
            return frozenset(items)

        ordered_module_paths: list[str] = []
        seen: set[str] = set()
        for part in model_parts:
            for mname, _ in part.named_modules():
                if mname not in seen:
                    seen.add(mname)
                    ordered_module_paths.append(mname)
            break  # Structure is identical across parts.

        children_by_parent: dict[str, list[str]] = defaultdict(list)
        for mpath in ordered_module_paths:
            if mpath:
                parent = ".".join(mpath.split(".")[:-1])
                children_by_parent[parent].append(mpath)

        collapse_count: dict[str, int] = {}
        directly_skipped: set[str] = set()

        for children in children_by_parent.values():
            if len(children) < 2:
                continue
            by_sig: dict[frozenset, list[str]] = defaultdict(list)
            for child in children:
                by_sig[_sig(child)].append(child)
            for group in by_sig.values():
                if len(group) >= 2:
                    collapse_count[group[0]] = len(group)
                    directly_skipped.update(group[1:])

        all_skipped: set[str] = set()
        for mpath in ordered_module_paths:
            if mpath in directly_skipped:
                all_skipped.add(mpath)
                continue
            for skip in directly_skipped:
                if mpath.startswith(skip + "."):
                    all_skipped.add(mpath)
                    break

        C_PATH, C_SHAPE, C_PARAMS, C_PCT = 52, 22, 14, 9
        sep = "─" * (C_PATH + C_SHAPE + C_PARAMS + C_PCT)
        lines = [
            "Model parameter breakdown:",
            f"{'Module / Parameter':<{C_PATH}} {'Shape':<{C_SHAPE}} {'Params':>{C_PARAMS}} {'%':>{C_PCT}}",
            sep,
        ]

        for mpath in ordered_module_paths:
            if mpath in all_skipped:
                continue
            mtotal = _module_total(mpath)
            if mtotal == 0:
                continue

            depth = mpath.count(".") + 1 if mpath else 0
            mname = mpath.split(".")[-1] if mpath else "(model)"
            n = collapse_count.get(mpath, 1)
            if n > 1:
                mname = f"{mname} [×{n}]"
            indent = "  " * depth
            pct_str = f"{100.0 * mtotal / grand_total:.2f}%" if depth <= 1 else ""

            label = f"{indent}{mname}"
            lines.append(
                f"{label:<{C_PATH}} {'':>{C_SHAPE}} {mtotal:>{C_PARAMS},} {pct_str:>{C_PCT}}"
            )

            param_indent = "  " * (depth + 1)
            for ppath, (shape, numel, trainable) in all_params.items():
                pparts = ppath.split(".")
                if ".".join(pparts[:-1]) != mpath:
                    continue
                pname = pparts[-1]
                tr_flag = "" if trainable else " (frozen)"
                plabel = f"{param_indent}.{pname}{tr_flag}"
                lines.append(
                    f"{plabel:<{C_PATH}} {str(shape):<{C_SHAPE}} {numel:>{C_PARAMS},} {'':>{C_PCT}}"
                )

        lines += [
            sep,
            f"{'TOTAL':<{C_PATH}} {'':>{C_SHAPE}} {grand_total:>{C_PARAMS},} {'100.00%':>{C_PCT}}",
        ]
        return "\n".join(lines)

    def _log_param_table(self, model_parts) -> None:
        """Log parameter table to training log and TensorBoard (rank 0 only)."""
        import torch.distributed as dist
        from torchtitan.tools.logging import logger

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        table = self._build_param_table(model_parts)
        logger.info("\n" + table)

        for writer in self._collect_writers():
            try:
                writer.add_text("model/param_breakdown", f"```\n{table}\n```", 0)
            except Exception as exc:
                logger.warning(f"Failed to log param table to TensorBoard: {exc}")

    def _log_config_to_tensorboard(self, job_config: JobConfig) -> None:
        """Log job config as a TensorBoard text entry (rank 0 only)."""
        import torch.distributed as dist
        from torchtitan.tools.logging import logger

        if not dist.is_initialized() or dist.get_rank() != 0:
            return

        writers = self._collect_writers()
        if not writers:
            return

        try:
            config_dict = self._filter_none_recursive(job_config.to_dict())
            if tomli_w is not None:
                buf = io.BytesIO()
                tomli_w.dump(config_dict, buf)
                text = buf.getvalue().decode("utf-8")
            else:
                text = json.dumps(config_dict, indent=2, sort_keys=True)
            for writer in writers:
                writer.add_text("config", text, 0)
        except Exception as exc:
            logger.warning(f"Failed to log config to TensorBoard: {exc}")

    @property
    def model_parts(self):
        return getattr(self, "_model_parts", None)

    @model_parts.setter
    def model_parts(self, value):
        self._model_parts = value
        if value is not None:
            self._log_parameter_count(value)
            self._nparams_total = sum(p.numel() for part in value for p in part.parameters())
            self._nparams_trainable = sum(
                p.numel() for part in value for p in part.parameters() if p.requires_grad
            )
            try:
                self._log_param_table(value)
            except Exception as exc:
                from torchtitan.tools.logging import logger
                logger.warning(f"Failed to build parameter table: {exc}")
            if self.param_logger:
                self.param_logger.update_model_parts(value)
                self.param_logger.enable_model_activation_logging()

    def _log_parameter_count(self, model_parts):
        """Log total and trainable parameter counts to TensorBoard at step 0."""
        total_params = 0
        trainable_params = 0
        for model in model_parts:
            for p in model.parameters():
                total_params += p.numel()
                if p.requires_grad:
                    trainable_params += p.numel()

        metrics = {
            "model_info/total_parameters": total_params,
            "model_info/trainable_parameters": trainable_params,
        }
        self.logger.log(metrics, step=0)

        from torchtitan.tools.logging import logger

        color = self.color
        logger.info(
            f"{color.cyan}Model parameters: {total_params:,} total, {trainable_params:,} trainable{color.reset}"
        )

    def __setattr__(self, name, value):
        if name == "optimizers":
            super().__setattr__(name, value)

            # Grab model_parts and loss_fn from the calling trainer, which runs
            # `self.metrics_processor.optimizers = self.optimizers`.
            import inspect

            caller_frame = inspect.currentframe().f_back
            if "self" in caller_frame.f_locals:
                trainer = caller_frame.f_locals["self"]

                if hasattr(self, "param_logger") and self.param_logger and not self.param_logger.model_parts:
                    if hasattr(trainer, "model_parts") and trainer.model_parts:
                        self.param_logger.update_model_parts(trainer.model_parts)
                        self.param_logger.enable_model_activation_logging()
                        from torchtitan.tools.logging import logger

                        logger.info(
                            f"Parameter logging: Connected to {len(trainer.model_parts)} model parts "
                            "via optimizers setter"
                        )

                if hasattr(trainer, "loss_fn") and trainer.loss_fn is not None:
                    loss_fn = trainer.loss_fn
                    # Unwrap RescaleAccumulatedLoss.
                    if hasattr(loss_fn, "unwrapped_loss_fn"):
                        loss_fn = loss_fn.unwrapped_loss_fn
                    if hasattr(loss_fn, "get_last_metrics"):
                        self._early_exit_loss_fn = loss_fn

            if hasattr(self, "param_logger") and self.param_logger:
                self.param_logger.update_optimizers(value)
        elif name == "lr_schedulers":
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def _get_current_learning_rate(self) -> float | None:
        """Extract current learning rate from the first optimizer."""
        try:
            if self.optimizers and len(self.optimizers) > 0:
                first_optimizer = next(iter(self.optimizers))
                if hasattr(first_optimizer, "param_groups") and first_optimizer.param_groups:
                    return first_optimizer.param_groups[0].get("lr")
        except (AttributeError, IndexError, KeyError, StopIteration):
            pass
        return None

    def _calculate_perplexity(self, loss: float) -> float | None:
        """Calculate perplexity from loss, handling potential overflow."""
        try:
            if loss > 100:
                return None
            return math.exp(loss)
        except (OverflowError, ValueError):
            return None

    def _collect_expert_load_metrics(self, step: int) -> dict[str, float]:
        """Collect MoE expert load stats over the window since the previous call.

        Rank 0 only; with EP > 1 the stats cover the local expert shard only.
        """
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return {}
        if not self._model_parts:
            return {}

        try:
            from torchtitan.models.moe import MoE
        except ImportError:
            return {}

        # titan_sci's MoE is a separate copy, not a subclass of torchtitan's.
        try:
            from titan_sci.models.moe import MoE as SciMoE

            moe_classes: tuple = (MoE, SciMoE)
        except ImportError:
            moe_classes = (MoE,)

        total_delta: torch.Tensor | None = None
        num_moe_layers = 0
        per_layer_stats: list[tuple[str, torch.Tensor]] = []

        for model_part in self._model_parts:
            layers = getattr(model_part, "layers", None)
            if layers is None:
                continue
            for layer_id, layer in layers.items():
                inner = getattr(layer, "_checkpoint_wrapped_module", layer)
                # Models put the MoE under either "feed_forward" or "moe".
                ff = None
                for attr in ("feed_forward", "moe", ""):
                    candidate = getattr(inner, attr, None)
                    if isinstance(candidate, moe_classes):
                        ff = candidate
                        break
                if ff is None:
                    continue
                tpe = getattr(ff, "tokens_per_expert", None)
                if tpe is None:
                    continue

                current = tpe.detach().float()
                key = f"{id(model_part)}_{layer_id}"
                prev = self._tpe_snapshot.get(key)
                delta = (current - prev) if prev is not None else current.clone()
                self._tpe_snapshot[key] = current.clone()

                layer_total = delta.sum().item()
                if layer_total > 0:
                    per_layer_stats.append((f"moe/expert_usage/layer_{layer_id}", delta / layer_total))

                total_delta = delta if total_delta is None else total_delta + delta
                num_moe_layers += 1

            shared_ff = getattr(model_part, "shared_feed_forward", None)
            if shared_ff is None:
                continue

            inner = getattr(shared_ff, "_checkpoint_wrapped_module", shared_ff)
            ff = None
            for attr in ("feed_forward", "moe", ""):
                candidate = getattr(inner, attr, None)
                if isinstance(candidate, moe_classes):
                    ff = candidate
                    break
            if ff is None:
                continue

            # titan_sci MoE only: [n_layers, n_experts] routing counts per call site.
            per_layer_tpe = getattr(ff, "per_layer_tokens_per_expert", None)
            if per_layer_tpe is not None:
                current_pl = per_layer_tpe.detach().float()
                for i in range(current_pl.shape[0]):
                    key_pl = f"{id(model_part)}_shared_pl_{i}"
                    prev_pl = self._tpe_snapshot.get(key_pl)
                    row = current_pl[i]
                    delta_pl = (row - prev_pl) if prev_pl is not None else row.clone()
                    self._tpe_snapshot[key_pl] = row.clone()
                    pl_total = delta_pl.sum().item()
                    if pl_total > 0:
                        per_layer_stats.append((f"moe/expert_usage/shared_layer_{i}", delta_pl / pl_total))

            tpe = getattr(ff, "tokens_per_expert", None)
            if tpe is None:
                continue

            current = tpe.detach().float()
            key = f"{id(model_part)}_shared"
            prev = self._tpe_snapshot.get(key)
            delta = (current - prev) if prev is not None else current.clone()
            self._tpe_snapshot[key] = current.clone()

            total_delta = delta if total_delta is None else total_delta + delta
            num_moe_layers += 1

        if total_delta is None or num_moe_layers == 0:
            return {}

        avg = total_delta / num_moe_layers
        total = avg.sum().item()
        if total <= 0:
            return {}

        fracs = avg / total
        n = fracs.numel()
        imbalance = (fracs.max() * n).item()
        mean_frac = 1.0 / n
        entropy = -(fracs * fracs.clamp(min=1e-10).log()).sum().item()
        max_entropy = math.log(n)

        # Per-expert block character; each level is 0.25x of the mean load.
        _bars = " ▁▂▃▄▅▆▇█"

        def _bar_char(f: float) -> str:
            ratio = f / mean_frac  # 1.0 = perfectly balanced
            idx = min(8, max(0, int(ratio * 4)))
            return _bars[idx]

        bar = "".join(_bar_char(f.item()) for f in fracs)

        from torchtitan.tools.logging import logger as titan_logger

        titan_logger.info(
            f"[ExpertLoad step={step}] n_experts(local)={n} over {num_moe_layers} layers | "
            f"imbalance={imbalance:.3f}x  entropy={100 * entropy / max_entropy:.1f}%  "
            f"min={fracs.min().item():.4f}  max={fracs.max().item():.4f}"
            f"\n  per-expert (▁=low █=high, rel. to mean): {bar}"
        )

        # Pseudo-histograms: repeat each expert index proportionally to its load so that
        # bins=n_experts yields one bar per expert.
        for writer in self._collect_writers():
            try:
                import numpy as np
                for tag, layer_fracs in per_layer_stats:
                    n_e = layer_fracs.numel()
                    counts = np.round(layer_fracs.cpu().numpy() * 10000).astype(np.int32)
                    samples = np.repeat(np.arange(n_e, dtype=np.float32), counts)
                    if len(samples) > 0:
                        writer.add_histogram(tag, samples, global_step=step, bins=n_e)
                counts = np.round(fracs.cpu().numpy() * 10000).astype(np.int32)
                samples = np.repeat(np.arange(n, dtype=np.float32), counts)
                writer.add_histogram("moe/expert_usage", samples, global_step=step, bins=n)
            except Exception as exc:
                titan_logger.warning(f"Failed to log expert usage histogram: {exc}")
                break

        return {
            "moe/load_imbalance": imbalance,
            "moe/load_entropy_pct": 100.0 * entropy / max_entropy if max_entropy > 0 else 0.0,
            "moe/load_min_frac": fracs.min().item(),
            "moe/load_max_frac": fracs.max().item(),
        }

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        """Log with parameter statistics, learning rate, perplexity and grad norm added."""
        combined_extra_metrics = {}

        if extra_metrics:
            combined_extra_metrics.update(extra_metrics)

        if grad_norm is not None:
            combined_extra_metrics["grad_norm"] = grad_norm

        current_lr = self._get_current_learning_rate()
        if current_lr is not None:
            combined_extra_metrics["learning_rate"] = current_lr

        perplexity = self._calculate_perplexity(global_avg_loss)
        if perplexity is not None:
            combined_extra_metrics["loss_metrics/global/avg_ppl"] = perplexity

        if hasattr(self, "optimizers") and self.optimizers:
            try:
                first_opt = next(iter(self.optimizers), None)
                if first_opt and hasattr(first_opt, "normalizer"):
                    rounding_state = first_opt.normalizer.get_rounding_state()
                    if "int_bits" in rounding_state:
                        combined_extra_metrics["rounding/int_bits"] = rounding_state["int_bits"]
                    if "exponent" in rounding_state:
                        combined_extra_metrics["rounding/exponent"] = rounding_state["exponent"]
            except (StopIteration, AttributeError):
                pass

        if self.param_logger and self.param_logger.should_log(step):
            param_stats = self.param_logger.compute_stats(step)
            if param_stats:
                combined_extra_metrics.update(param_stats)
                from torchtitan.tools.logging import logger
                logger.info(f"Step {step}: Added {len(param_stats)} parameter statistics to metrics")

        super().log(step, global_avg_loss, global_max_loss, grad_norm, combined_extra_metrics or None)

        if self._model_parts:
            for model in self._model_parts:
                val = getattr(model, "_logits_norm", None)
                if val is not None:
                    combined_extra_metrics["logits_norm"] = val
                    break

        if self._nparams_total is not None:
            combined_extra_metrics["model/total_params_M"] = self._nparams_total / 1e6
        if self._nparams_trainable is not None and self._nparams_trainable != self._nparams_total:
            combined_extra_metrics["model/trainable_params_M"] = self._nparams_trainable / 1e6


def build_metrics_processor_with_parameter_logging(
    job_config: BaseJobConfig,
    parallel_dims: ParallelDims,
    model_args=None,
    tag: str | None = None,
) -> MetricsProcessor:
    """Drop-in replacement for torchtitan's build_metrics_processor with parameter logging."""
    if hasattr(job_config, 'parameter_logging'):
        return EnhancedMetricsProcessor(job_config, parallel_dims, tag)
    else:
        return build_metrics_processor(job_config, parallel_dims, model_args, tag)


build_enhanced_metrics_processor = build_metrics_processor_with_parameter_logging