from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import BaseValidator
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from src.rasr.constants import IGNORE_INDEX
from src.rasr.data.dataloader import build_rasr_validation_dataloader
from src.rasr.data.tokenization import EquationTokenizer


def _as_scalar_loss(loss_output) -> torch.Tensor:
    if isinstance(loss_output, tuple):
        # Keep compatibility with old and custom loss signatures.
        if len(loss_output) >= 2 and isinstance(loss_output[1], torch.Tensor) and loss_output[1].ndim == 0:
            return loss_output[1]
        return loss_output[0]
    return loss_output


class RASRValidator(BaseValidator):
    """Teacher-forcing validation for RASR Arrow datasets."""

    def __init__(
        self,
        job_config,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None] | None = None,
        maybe_enable_amp: Generator[None, None, None] | None = None,
        metrics_processor: MetricsProcessor | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
    ):
        super().__init__(job_config)
        _ = tokenizer, pp_schedule, pp_has_first_stage, pp_has_last_stage
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor

        self._validation_dataloader_config = {
            "dp_world_size": dp_world_size,
            "dp_rank": dp_rank,
            "tokenizer": None,
            "job_config": job_config,
        }
        self.validation_dataloader = build_rasr_validation_dataloader(**self._validation_dataloader_config)

        eq_tokenizer = EquationTokenizer.from_config(self.job_config.tokenizer)
        token_to_index = eq_tokenizer.token_to_index
        if "0" in token_to_index and "9" in token_to_index:
            self.digit_start_id = token_to_index["0"]
            self.digit_end_id = token_to_index["9"]
        else:
            self.digit_start_id = None
            self.digit_end_id = None
        val_set_name = job_config.dataloader.val_set_name

        logger.info(
            f"RASRValidator: enabled with freq={job_config.validation.freq}, "
            f"set_name={val_set_name}"
        )

    def should_validate(self, step: int) -> bool:
        return step % self.job_config.validation.freq == 0 or step == self.job_config.training.steps

    def _reset_validation_dataloader(self) -> None:
        self.validation_dataloader = build_rasr_validation_dataloader(**self._validation_dataloader_config)

    @staticmethod
    def _move_batch_to_device(input_dict: dict[str, torch.Tensor], labels: torch.Tensor, device: torch.device):
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return input_dict, labels

    @staticmethod
    def _sync_sum(value: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        if not model_parts:
            logger.warning("RASRValidator: no model parts available, skipping validation.")
            return
        if self.parallel_dims.pp_enabled:
            logger.warning("RASRValidator currently supports non-pipeline execution only. Skipping validation.")
            return

        self._reset_validation_dataloader()
        model = model_parts[0]
        device = next(model.parameters()).device

        old_mode = model.training
        model.eval()

        stats = defaultdict(lambda: torch.tensor(0.0, device=device))
        max_steps = int(getattr(self.job_config.validation, "steps", -1))
        val_iterator = iter(self.validation_dataloader)

        with torch.no_grad():
            step_idx = 0
            while True:
                if max_steps > 0 and step_idx >= max_steps:
                    break
                try:
                    input_dict, labels = next(val_iterator)
                except StopIteration:
                    break

                input_dict, labels = self._move_batch_to_device(input_dict, labels, device)
                context_1 = (
                    self.validation_context()
                    if self.validation_context is not None
                    else nullcontext()
                )
                context_2 = self.maybe_enable_amp if self.maybe_enable_amp is not None else nullcontext()
                with context_1, context_2:
                    logits = model(
                        tbl=input_dict["input"],
                        decoder_input_ids=input_dict["decoder_input_ids"],
                        trg_len=input_dict["trg_len"],
                        tbl_mask=input_dict["tbl_mask"],
                        decoder_self_attn_mask=input_dict["decoder_self_attn_mask"],
                        decoder_cross_attn_mask=input_dict["decoder_cross_attn_mask"],
                    )
                    loss = _as_scalar_loss(self.loss_fn(logits, labels))

                predicted_tokens = logits.argmax(dim=-1)
                ignore_mask = labels != IGNORE_INDEX
                valid_token_count = ignore_mask.sum().to(torch.float32)
                sample_count = torch.tensor(float(input_dict["trg_len"].size(0)), device=device)

                correct_tokens = (predicted_tokens == labels) & ignore_mask
                batch_accuracy = correct_tokens.sum(dim=-1).to(torch.float32) / input_dict["trg_len"].to(torch.float32)
                solved = (batch_accuracy == 1.0).sum().to(torch.float32)

                if self.digit_start_id is not None:
                    is_digit_target = (labels >= self.digit_start_id) & (labels <= self.digit_end_id)
                    is_digit_pred = (predicted_tokens >= self.digit_start_id) & (predicted_tokens <= self.digit_end_id)
                    correct_skeleton = ((predicted_tokens == labels) | (is_digit_target & is_digit_pred)) & ignore_mask
                else:
                    correct_skeleton = (predicted_tokens == labels) & ignore_mask
                batch_skeleton_accuracy = correct_skeleton.sum(dim=-1).to(torch.float32) / input_dict["trg_len"].to(
                    torch.float32
                )
                solved_skeleton = (batch_skeleton_accuracy == 1.0).sum().to(torch.float32)

                stats["loss_weighted_sum"] += loss * valid_token_count
                stats["tokens"] += valid_token_count
                stats["samples"] += sample_count
                stats["accuracy_sum"] += batch_accuracy.sum()
                stats["solved_sum"] += solved
                stats["solved_skeleton_sum"] += solved_skeleton
                stats["num_batches"] += torch.tensor(1.0, device=device)

                step_idx += 1

        for key in list(stats.keys()):
            stats[key] = self._sync_sum(stats[key])

        if not dist.is_initialized() or dist.get_rank() == 0:
            tokens = max(stats["tokens"].item(), 1.0)
            samples = max(stats["samples"].item(), 1.0)
            mean_loss = stats["loss_weighted_sum"].item() / tokens
            ppl = float(torch.exp(stats["loss_weighted_sum"] / stats["tokens"]).item())
            accuracy = stats["accuracy_sum"].item() / samples
            solved = stats["solved_sum"].item() / samples
            solved_skeleton = stats["solved_skeleton_sum"].item() / samples

            logger.info(
                f"[RASR validation] step={step} "
                f"loss={mean_loss:.4f} ppl={ppl:.4f} "
                f"acc={accuracy:.4f} solved={solved:.4f} "
                f"solved_skeleton={solved_skeleton:.4f} "
                f"samples={int(samples)} tokens={int(tokens)}"
            )

            wandb_metrics = {
                "validation/rasr/loss": mean_loss,
                "validation/rasr/ppl": ppl,
                "validation/rasr/accuracy": accuracy,
                "validation/rasr/solved": solved,
                "validation/rasr/solved_skeleton": solved_skeleton,
            }

            if self.metrics_processor is not None and getattr(self.metrics_processor, "logger", None) is not None:
                self.metrics_processor.logger.log(wandb_metrics, step)

        model.train(old_mode)


def build_rasr_validator(
    job_config,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
) -> BaseValidator:
    return RASRValidator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )
