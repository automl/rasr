from __future__ import annotations

import copy
import math
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from accelerate.utils import gather, gather_object
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import SequentialSampler
from transformers import Trainer as HFTrainer
from trl import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.models import prepare_deepspeed, prepare_fsdp

try:
    from trl.models import unwrap_model_for_generation
except Exception:
    from trl.models.utils import unwrap_model_for_generation

from trl.trainer.utils import selective_log_softmax

from src.rasr.inference.checkpoint import save_weights
from src.rasr.masks import make_rasr_masks
from src.rasr.post_train.constants import N_VALIDATION_ROWS


def nanstd(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    variance = torch.nanmean((tensor - mean) ** 2, dim=dim, keepdim=keepdim)
    if dim is None:
        count = torch.sum(~torch.isnan(tensor))
    else:
        count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=keepdim)
    variance = variance * count / (count - 1)
    return torch.sqrt(variance)


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    permutation = torch.randperm(first_tensor.shape[0])
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}


def _log_comb(n: int, k: int) -> float:
    """Log of C(n, k) using lgamma; returns -inf when k < 0 or k > n."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _oracle_at_k(test_scores: np.ndarray, k: int) -> float:
    """Exact expected best test score in a random k-subset: Σ_i test[σ(i)]·C(n-i-1, k-1)/C(n, k), σ sorting descending."""
    n = len(test_scores)
    if n == 0 or k <= 0:
        return float("nan")
    k = min(k, n)
    order = np.argsort(test_scores)[::-1]
    log_cn_k = _log_comb(n, k)
    result = 0.0
    for i in range(n):
        remaining = n - i - 1
        if remaining < k - 1:
            break
        result += math.exp(_log_comb(remaining, k - 1) - log_cn_k) * float(test_scores[order[i]])
    return result


_SELECT_K_VALUES: list[int] = [1, 4, 16, 32, 64, 128, 256]
_CW_GRID: list[float] = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
_N_MC_SAMPLES: int = 500


def _adaptive_select_at_k(
    val_r2: np.ndarray,
    sel_r2: np.ndarray,
    complexity_rewards: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo select@k, picking per subset the complexity weight that maximises the selected sel_r2."""
    n = len(val_r2)
    k = min(k, n)
    if n == 0:
        return float("nan")

    cw_arr = np.asarray(_CW_GRID, dtype=np.float64)  # (C,)
    sel_scores = sel_r2[:, None] + cw_arr[None, :] * complexity_rewards[:, None]  # (n, C)

    if k == n:
        best_in_full = np.argmax(sel_scores, axis=0)          # (C,)
        sel_r2_per_cw = sel_r2[best_in_full]                  # (C,)
        best_cw = int(np.argmax(sel_r2_per_cw))
        return float(val_r2[best_in_full[best_cw]])

    idx_mc = np.argsort(rng.random((_N_MC_SAMPLES, n)), axis=1)[:, :k]  # (n_mc, k)

    sub_sel = sel_scores[idx_mc]     # (n_mc, k, C)
    sub_sel_r2 = sel_r2[idx_mc]     # (n_mc, k)
    sub_val_r2 = val_r2[idx_mc]     # (n_mc, k)

    best_in_sub = np.argmax(sub_sel, axis=1)        # (n_mc, C)
    rows = np.arange(_N_MC_SAMPLES)
    sel_r2_per_cw = sub_sel_r2[rows[:, None], best_in_sub]   # (n_mc, C)
    best_cw = np.argmax(sel_r2_per_cw, axis=1)               # (n_mc,)
    selected_in_sub = best_in_sub[rows, best_cw]              # (n_mc,)
    selected_val = sub_val_r2[rows, selected_in_sub]          # (n_mc,)

    return float(np.mean(selected_val))


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        model_config_path: str | None = None,
        rs_grpo_beta: float = 0.0,
        eval_generation_chunk_size: int = 32,
        train_generation_chunk_size: int,
        offload_ref_model: bool = False,
        **kwargs,
    ):
        # TRL's GRPOTrainer currently does not accept `ref_model` directly.
        ref_model = kwargs.pop("ref_model", None)
        if ref_model is None:
            ref_model = kwargs.pop("reference_model", None)

        # With beta > 0, TRL auto-builds a ref model from the HF Hub id, which the RASR wrapper
        # lacks. Force beta=0 during super().__init__() to skip that, then restore it.
        trl_args = kwargs.get("args")
        if trl_args is None and args:
            trl_args = args[2] if len(args) > 2 else None
        original_beta = None
        if trl_args is not None and ref_model is not None:
            original_beta = trl_args.beta
            trl_args.beta = 0.0

        super().__init__(*args, **kwargs)

        if original_beta is not None:
            trl_args.beta = original_beta
            self.beta = original_beta

        self._offload_ref_model = offload_ref_model
        self.ref_model = ref_model
        if self.ref_model is not None:
            if offload_ref_model:
                # Logp batches are moved to CPU per chunk during generation.
                self.ref_model = self.ref_model.cpu()
            elif self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.rs_grpo_beta = float(rs_grpo_beta)
        self._eval_generation_chunk_size = int(eval_generation_chunk_size)
        self._train_generation_chunk_size = int(train_generation_chunk_size)
        self.model_config_path = Path(model_config_path) if model_config_path else None
        self._buffered_inputs = None
        self._step = 0
        self._eval_r2_scores: list[float] = []
        self._eval_max_r2_scores: list[float] = []
        self._eval_train_max_r2_scores: list[float] = []
        self._eval_select_at_k: dict[int, list[float]] = {k: [] for k in _SELECT_K_VALUES}
        self._eval_oracle_at_k: dict[int, list[float]] = {k: [] for k in _SELECT_K_VALUES}
        self._mc_rng = np.random.default_rng(getattr(self.args, "seed", None))

    def _get_eval_sampler(self, eval_dataset):
        # TRL's RepeatSampler needs global_eval_batch_size >= num_generations_eval. Instead, all eval
        # completions per prompt are generated via num_return_sequences in _generate_and_score_completions.
        return SequentialSampler(eval_dataset)

    def _build_generation_samples(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> list[dict[str, Union[torch.Tensor, Any]]]:
        generation_samples: list[dict[str, Union[torch.Tensor, Any]]] = []
        for sample in inputs:
            sample_tbl = sample["tbl"]
            if not isinstance(sample_tbl, torch.Tensor):
                raise TypeError("Expected sample['tbl'] to be a torch.Tensor.")

            generation_tbl = sample_tbl[:-N_VALIDATION_ROWS]
            generation_sample = dict(sample)
            generation_sample["tbl"] = generation_tbl
            generation_samples.append(generation_sample)

        return generation_samples

    @profiling_decorator
    def _get_per_token_logps(
        self, model, tbl, completion_ids, tbl_mask, batch_size=None, cpu_offload: bool = False
    ) -> torch.Tensor:
        batch_size = batch_size or tbl.size(0)
        src_device = tbl.device

        all_logps = []
        for i in range(0, tbl.size(0), batch_size):
            tbl_batch = tbl[i : i + batch_size]
            tbl_mask_batch = tbl_mask[i : i + batch_size]
            decoder_input_ids_batch = completion_ids[i : i + batch_size, :-1]
            target_ids_batch = completion_ids[i : i + batch_size, 1:]

            if cpu_offload:
                tbl_batch = tbl_batch.cpu()
                tbl_mask_batch = tbl_mask_batch.cpu()
                decoder_input_ids_batch = decoder_input_ids_batch.cpu()
                target_ids_batch = target_ids_batch.cpu()

            trg_len = torch.full(
                (decoder_input_ids_batch.size(0),),
                decoder_input_ids_batch.size(1),
                dtype=torch.long,
                device=decoder_input_ids_batch.device,
            )
            key_is_valid = tbl_mask_batch[:, :, 0]
            decoder_self_attn_mask, decoder_cross_attn_mask = make_rasr_masks(
                key_is_valid=key_is_valid,
                seq_len=decoder_input_ids_batch.size(1),
                trg_len=trg_len,
            )

            logits = model(
                tbl=tbl_batch,
                decoder_input_ids=decoder_input_ids_batch,
                trg_len=trg_len,
                decoder_self_attn_mask=decoder_self_attn_mask,
                decoder_cross_attn_mask=decoder_cross_attn_mask,
                tbl_mask=tbl_mask_batch,
            ).logits
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, target_ids_batch)
            if cpu_offload:
                logps = logps.to(src_device)
            all_logps.append(logps)

        return torch.cat(all_logps, dim=0)

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, Union[torch.Tensor, Any]]):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(completion_ids_list), len(self.reward_funcs), device=device)
        reward_extras: dict[str, list[float]] = {}

        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state
        reward_kwargs["log_extra"] = self._log_completion_extra
        reward_kwargs["log_metric"] = self._log_metric
        prompts = [example.get("prompt", "") for example in inputs]

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                raise NotImplementedError("Module reward functions are not supported in RASR post-training.")
            output = reward_func(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_ids_list,
                **reward_kwargs,
            )
            if isinstance(output, dict):
                reward_values = output["reward"]
                for k, v in output.items():
                    if k != "reward":
                        reward_extras[f"{self.reward_func_names[i]}/{k}"] = v
            else:
                reward_values = output
            reward_values = [r if r is not None else torch.nan for r in reward_values]
            rewards_per_func[:, i] = torch.tensor(reward_values, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func, reward_extras

    def _tlog(self, msg: str) -> None:
        rank = self.accelerator.process_index
        print(f"[rank{rank}][TIMING] {msg}", file=sys.stderr, flush=True)

    @profiling_decorator
    def _generate_and_score_completions(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        generation_samples = self._build_generation_samples(inputs)
        tbl_inputs = self.processing_class(generation_samples, return_tensors="pt", padding=True, padding_side="left")
        tbl_inputs = HFTrainer._prepare_inputs(self, tbl_inputs)
        tbl = tbl_inputs["tbl"]
        tbl_mask = tbl_inputs["tbl_mask"]

        t0 = time.perf_counter()
        self._tlog(f"generation start  step={self.state.global_step} mode={mode} batch={tbl.shape[0]}")
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            with FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext():
                if mode == "eval":
                    # Generate num_generations_eval candidates per prompt in chunks to avoid OOM.
                    n = self.num_generations_eval
                    chunk_size = self._eval_generation_chunk_size
                    B = tbl.shape[0]
                    eval_gen_config = copy.copy(self.generation_config)
                    chunks: list[torch.Tensor] = []
                    for start in range(0, n, chunk_size):
                        eval_gen_config.num_return_sequences = min(chunk_size, n - start)
                        chunk_ids = unwrapped_model.generate(tbl=tbl, generation_config=eval_gen_config)
                        chunks.append(chunk_ids.view(B, eval_gen_config.num_return_sequences, -1))
                    # (B, n, L) -> (B*n, L), keeping completions grouped by prompt.
                    completion_ids = torch.cat(chunks, dim=1).view(B * n, -1)
                else:
                    # TRL repeats inputs via the DataLoader, so one completion per table; chunk to cap memory.
                    chunk_size = self._train_generation_chunk_size
                    B = tbl.shape[0]
                    train_chunks: list[torch.Tensor] = []
                    for start in range(0, B, chunk_size):
                        tbl_batch = tbl[start : start + chunk_size]
                        chunk_ids = unwrapped_model.generate(tbl=tbl_batch, generation_config=self.generation_config)
                        train_chunks.append(chunk_ids)
                    completion_ids = torch.cat(train_chunks, dim=0)
        self._tlog(f"generation done   {time.perf_counter() - t0:.1f}s  completion_ids={completion_ids.shape}")

        if mode == "eval":
            # Align tbl/inputs with the (B*n) completion_ids.
            tbl = tbl.repeat_interleave(n, dim=0)
            tbl_mask = tbl_mask.repeat_interleave(n, dim=0)
            inputs = [inp for inp in inputs for _ in range(n)]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [token.item() for token, mask in zip(row, mask_row) if mask]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)
        self._tlog(
            f"completion lengths min={completion_lengths.min().item()} max={completion_lengths.max().item()} "
            f"mean={completion_lengths.float().mean().item():.1f}"
        )

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            if mode != "eval" and (
                self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps
            ):
                t1 = time.perf_counter()
                self._tlog("old_logps start")
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    tbl,
                    completion_ids,
                    tbl_mask,
                    batch_size,
                )
                self._tlog(f"old_logps done    {time.perf_counter() - t1:.1f}s")
            else:
                old_per_token_logps = None

            if mode != "eval" and self.beta != 0.0:
                t2 = time.perf_counter()
                self._tlog("ref_logps start")
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        tbl,
                        completion_ids,
                        tbl_mask,
                        batch_size,
                        cpu_offload=self._offload_ref_model,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            tbl,
                            completion_ids,
                            tbl_mask,
                            batch_size,
                        )
                self._tlog(f"ref_logps done    {time.perf_counter() - t2:.1f}s")
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids[:, 1:], skip_special_tokens=True)
        t3 = time.perf_counter()
        self._tlog("rewards start")
        rewards_per_func, reward_extras = self._calculate_rewards(inputs, completions_text, completion_ids_list)
        self._tlog(f"rewards done      {time.perf_counter() - t3:.1f}s")
        num_generations = (
            self.num_generations if mode == "train" else (self.args.num_generations_eval or self.num_generations)
        )

        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. "
                    "Must be one of 'batch', 'group', or 'none'."
                )

            if self.rs_grpo_beta > 0.0:
                # RS-GRPO advantage (Eq. 8): Â_β(yi) = (1/β)·(exp(β·r(yi)) / mean(exp(β·r(yj))) - 1).
                rewards_grouped = rewards.view(-1, num_generations)
                exp_r = torch.exp(self.rs_grpo_beta * rewards_grouped)
                advantages_grouped = (exp_r / exp_r.mean(dim=1, keepdim=True) - 1.0) / self.rs_grpo_beta
                advantages = advantages_grouped.view(-1)
            else:
                advantages = rewards - mean_grouped_rewards
                if self.scale_rewards != "none":
                    advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            if self.rs_grpo_beta > 0.0:
                reward_k = (grouped - mean_k) / (std_k + 1e-4)
            else:
                reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            if self.rs_grpo_beta > 0.0:
                rewards_grouped = rewards.view(-1, num_generations)
                exp_r = torch.exp(self.rs_grpo_beta * rewards_grouped)
                advantages_grouped = (exp_r / exp_r.mean(dim=1, keepdim=True) - 1.0) / self.rs_grpo_beta
                advantages = advantages_grouped.view(-1)
            else:
                advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        process_slice = slice(
            self.accelerator.process_index * len(tbl),
            (self.accelerator.process_index + 1) * len(tbl),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(completion_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)

        if mode == "eval":
            val_r2_raw = reward_extras.get("nrmse_reward/val_r2")
            if val_r2_raw:
                r2_scores = torch.tensor(val_r2_raw, dtype=torch.float32, device=device)
                r2_scores = gather(r2_scores)
                self._metrics[mode]["rewards/r2/val_mean"].append(torch.nanmean(r2_scores).item())
                grouped_r2 = r2_scores.view(-1, num_generations)
                max_r2 = grouped_r2.nan_to_num(nan=float("-inf")).amax(dim=1)
                max_r2[torch.isnan(grouped_r2).all(dim=1)] = float("nan")
                self._metrics[mode]["rewards/r2/mean_max"].append(torch.nanmean(max_r2).item())
                self._eval_r2_scores.extend(r2_scores[~torch.isnan(r2_scores)].tolist())
                self._eval_max_r2_scores.extend(max_r2[~torch.isnan(max_r2)].tolist())

            train_r2_raw = reward_extras.get("nrmse_reward/train_r2")
            if train_r2_raw:
                train_r2_tensor = torch.tensor(train_r2_raw, dtype=torch.float32, device=device)
                train_r2_tensor = gather(train_r2_tensor)
                self._metrics[mode]["rewards/r2/train_mean"].append(torch.nanmean(train_r2_tensor).item())
                grouped_train_r2 = train_r2_tensor.view(-1, num_generations)
                max_train_r2 = grouped_train_r2.nan_to_num(nan=float("-inf")).amax(dim=1)
                max_train_r2[torch.isnan(grouped_train_r2).all(dim=1)] = float("nan")
                self._metrics[mode]["rewards/r2/train_mean_max"].append(torch.nanmean(max_train_r2).item())
                self._eval_train_max_r2_scores.extend(max_train_r2[~torch.isnan(max_train_r2)].tolist())

            sel_r2_raw = reward_extras.get("nrmse_reward/sel_r2")
            if sel_r2_raw and val_r2_raw:
                sel_r2_tensor = torch.tensor(sel_r2_raw, dtype=torch.float32, device=device)
                sel_r2_tensor = gather(sel_r2_tensor)

                complexity_idx = next(
                    (i for i, name in enumerate(self.reward_func_names) if name == "complexity_reward"),
                    None,
                )
                complexity_gathered = (
                    rewards_per_func[:, complexity_idx] if complexity_idx is not None
                    else torch.zeros_like(sel_r2_tensor)
                )

                val_r2_2d = r2_scores.view(-1, num_generations).cpu().numpy()
                sel_r2_2d = sel_r2_tensor.view(-1, num_generations).cpu().numpy()
                complexity_2d = complexity_gathered.view(-1, num_generations).cpu().numpy()
                k_values = [k for k in _SELECT_K_VALUES if k <= num_generations]

                for prompt_val, prompt_sel_r2, prompt_complexity in zip(val_r2_2d, sel_r2_2d, complexity_2d):
                    valid = np.isfinite(prompt_val) & np.isfinite(prompt_sel_r2) & np.isfinite(prompt_complexity)
                    v = prompt_val[valid]
                    sr2 = prompt_sel_r2[valid]
                    c = prompt_complexity[valid]
                    if len(v) == 0:
                        continue
                    for k in k_values:
                        oracle = _oracle_at_k(v, k)
                        select = _adaptive_select_at_k(v, sr2, c, k, self._mc_rng)
                        if np.isfinite(oracle):
                            self._eval_oracle_at_k[k].append(oracle)
                        if np.isfinite(select):
                            self._eval_select_at_k[k].append(select)

        rewards_for_metrics = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards_for_metrics.mean().item())
        self._metrics[mode]["reward_std"].append(rewards_for_metrics.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["prompt"].extend([""] * len(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "tbl": tbl,
            "tbl_mask": tbl_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPO trainer does not support return_outputs=True.")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        tbl = inputs["tbl"]
        tbl_mask = inputs["tbl_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        per_token_logps = self._get_per_token_logps(
            model, tbl, completion_ids, tbl_mask, self._train_generation_chunk_size
        )

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )

        if self.off_policy_mask_threshold is not None:
            off_policy_mask = GRPOTrainer.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=old_per_token_logps,
                mask=completion_mask[:, 1:],
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            completion_mask_trimmed = completion_mask[:, 1:]
            log_importance_weights = (log_ratio * completion_mask_trimmed).sum(-1) / completion_mask_trimmed.sum(
                -1
            ).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        if self.beta != 0.0:
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        if self.loss_type == "cispo":
            coef_2 = None
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            coef_2 = None
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        completion_mask = completion_mask[:, 1:]
        mode = "train" if self.model.training else "eval"
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / normalizer
        elif self.loss_type in ["dapo", "cispo"]:
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "luspo":
            loss = (per_token_loss * completion_mask.sum(1, keepdim=True)).mean()
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        def masked_mean(x):
            return (x * completion_mask).sum() / completion_mask.sum()

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            gathered_low_clip = self.accelerator.gather(masked_mean(is_low_clipped.float()))
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(masked_mean(is_high_clipped.float()))
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(masked_mean(is_region_clipped.float()))
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            gathered_cispo_clip_ratio = self.accelerator.gather(masked_mean(is_cispo_clipped.float()))
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Only generate and score for the reward metrics; the eval loss is uninformative and slow.
        self._prepare_inputs(inputs)
        return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs: dict[str, float], start_time=None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val)
            for key, val in self._metrics[mode].items()
            if val
        }
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}

        if mode == "eval" and self._eval_r2_scores:
            scores_arr = np.array(self._eval_r2_scores)
            for p in [10, 25, 50, 75, 90]:
                logs[f"eval_rewards/r2/p{p}"] = float(np.percentile(scores_arr, p))
            self._eval_r2_scores.clear()
        if mode == "eval" and self._eval_max_r2_scores:
            max_scores_arr = np.array(self._eval_max_r2_scores)
            for p in [10, 25, 50, 75, 90]:
                logs[f"eval_rewards/r2/max_p{p}"] = float(np.percentile(max_scores_arr, p))
            self._eval_max_r2_scores.clear()
        if mode == "eval" and self._eval_train_max_r2_scores:
            train_max_scores_arr = np.array(self._eval_train_max_r2_scores)
            for p in [10, 25, 50, 75, 90]:
                logs[f"eval_rewards/r2/train_max_p{p}"] = float(np.percentile(train_max_scores_arr, p))
            self._eval_train_max_r2_scores.clear()

        if mode == "eval":
            for k in _SELECT_K_VALUES:
                if self._eval_oracle_at_k[k]:
                    logs[f"eval_rewards/oracle_at_{k}"] = float(np.mean(self._eval_oracle_at_k[k]))
                    self._eval_oracle_at_k[k].clear()
                if self._eval_select_at_k[k]:
                    logs[f"eval_rewards/select_at_{k}"] = float(np.mean(self._eval_select_at_k[k]))
                    self._eval_select_at_k[k].clear()

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not self.is_world_process_zero():
            return

        step_dir = Path(self.args.output_dir) / f"step-{self.state.global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        policy = self.accelerator.unwrap_model(self.model)
        save_weights(policy.model, step_dir)

        if self.model_config_path is not None and self.model_config_path.exists():
            shutil.copy2(self.model_config_path, step_dir / "config.toml")

    def train(self, *args, **kwargs):
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        try:
            return super().train(*args, **kwargs)
        finally:
            for reward_fn in self.reward_funcs:
                close_fn = getattr(reward_fn, "close", None)
                if callable(close_fn):
                    close_fn()

