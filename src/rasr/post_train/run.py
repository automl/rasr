from __future__ import annotations

import argparse
import os
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

import tomli_w
import torch
from trl import GRPOConfig

from src.rasr.inference.checkpoint import load_rasr_model, save_weights
from src.rasr.post_train.config import PostTrainConfig
from src.rasr.post_train.dataset import RASRArrowMapDataset
from src.rasr.post_train.rewards import ComplexityRewardFn, NRMSERewardFn
from src.rasr.post_train.tokenizer_adapter import RASRProcessingClass
from src.rasr.post_train.trainer import CustomGRPOTrainer
from src.rasr.post_train.wrapper import RASRPolicyWrapper


class _GRPOConfig(GRPOConfig):
    def __post_init__(self):
        # TRL checks eval batch size divisibility by num_generations_eval assuming it counts
        # completions. Here it counts tables, so mask eval_strategy to skip that check.
        stored = self.eval_strategy
        self.eval_strategy = "no"
        super().__post_init__()
        self.eval_strategy = stored


def _default_model_config_path(model_path: Path) -> Path:
    config_path = model_path / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.toml under {model_path}.")
    return config_path


def _format_float(value: float) -> str:
    return f"{value:.2e}".replace(".00e", "e")


def _build_experiment_name(
    post_cfg: PostTrainConfig, model_path: Path, job_config: object, name: str | None = None
) -> str:
    flavor = getattr(getattr(job_config, "model", object()), "flavor", model_path.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"grpo_{name}_" if name else "grpo_"
    return (
        f"{prefix}{flavor}_"
        f"lr{_format_float(post_cfg.learning_rate)}_"
        f"bs{post_cfg.per_device_train_batch_size}_"
        f"g{post_cfg.num_generations}_"
        f"s{post_cfg.max_steps}_"
        f"cw{_format_float(post_cfg.reward.complexity_weight)}_"
        f"{timestamp}"
    )


def _build_output_dir(model_path: Path, config: PostTrainConfig, run_name: str) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    return model_path / "post_training" / run_name


def _apply_overrides(payload: dict, overrides: list[str]) -> None:
    it = iter(overrides)
    for key_str in it:
        if not key_str.startswith("--"):
            raise ValueError(f"Expected --key, got: {key_str!r}")
        val_str = next(it)
        parts = key_str[2:].split(".")
        d = payload
        for part in parts[:-1]:
            d = d[part]
        if val_str.lower() in ("true", "false"):
            val: object = val_str.lower() == "true"
        else:
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
        d[parts[-1]] = val


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run GRPO post-training for the RASR model.")
    parser.add_argument("--model_path", required=True, help="Path to pretrained run directory.")
    parser.add_argument("--config", required=True, help="Path to post-training TOML config.")
    parser.add_argument(
        "--model_config",
        default=None,
        help="Optional path to pretrained training config TOML. Defaults to <model_path>/config.toml.",
    )
    parser.add_argument("--step", type=int, default=None, help="Optional checkpoint step to load.")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu).")
    parser.add_argument("--name", default=None, help="Optional label inserted after 'grpo_' in the run name.")
    return parser.parse_known_args()


def main() -> None:
    args, overrides = parse_args()
    model_path = Path(args.model_path).resolve()
    model_config_path = (
        Path(args.model_config).resolve() if args.model_config else _default_model_config_path(model_path)
    )

    with Path(args.config).open("rb") as f:
        payload = tomllib.load(f)
    if overrides:
        _apply_overrides(payload, overrides)

    post_cfg = PostTrainConfig.from_payload(payload)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy_model, tokenizer, job_config = load_rasr_model(
        model_path=str(model_path),
        config_path=str(model_config_path),
        step=args.step,
        device=device,
        dtype=torch.bfloat16,
    )
    if post_cfg.gradient_checkpointing:
        policy_model.gradient_checkpointing = True

    if post_cfg.beta > 0.0:
        ref_model, _, _ = load_rasr_model(
            model_path=str(model_path),
            config_path=str(model_config_path),
            step=args.step,
            device=device,
            dtype=torch.bfloat16,
        )
        ref_model.eval()
        ref_model.requires_grad_(False)
    else:
        ref_model = None

    generated_run_name = _build_experiment_name(
        post_cfg=post_cfg, model_path=model_path, job_config=job_config, name=args.name
    )
    run_name = post_cfg.run_name or generated_run_name
    output_dir = _build_output_dir(model_path=model_path, config=post_cfg, run_name=run_name)
    # Keep run names and folder names aligned when output_dir is manually provided.
    run_name = output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "post_train.toml").open("wb") as f:
        tomli_w.dump(payload, f)

    os.environ["WANDB_PROJECT"] = "rasr-grpo"

    max_completion_length = int(post_cfg.max_completion_length)
    policy_wrapper = RASRPolicyWrapper(policy_model, tokenizer, max_length=max_completion_length)
    ref_wrapper = (
        RASRPolicyWrapper(ref_model, tokenizer, max_length=max_completion_length) if ref_model is not None else None
    )

    processing_class = RASRProcessingClass(tokenizer)

    dl_cfg = post_cfg.dataloader
    train_dataset = RASRArrowMapDataset(
        data_path=dl_cfg.data_path,
        set_name=dl_cfg.set_name,
        num_vars=dl_cfg.num_vars,
    )

    eval_dataset = None
    if dl_cfg.val_set_name:
        eval_dataset = RASRArrowMapDataset(
            data_path=dl_cfg.data_path,
            set_name=dl_cfg.val_set_name,
            num_vars=dl_cfg.num_vars,
        )

    nrmse_reward_fn = NRMSERewardFn(
        tokenizer=tokenizer,
        job_config=job_config,
        simplipy_engine_name=post_cfg.reward.simplipy_engine_name,
        refine_constants=post_cfg.reward.refine_constants,
        standardize_dataset=post_cfg.reward.standardize_dataset,
        n_restarts=post_cfg.reward.n_restarts,
        init_noise_scale=post_cfg.reward.init_noise_scale,
        refinement_timeout=post_cfg.reward.refinement_timeout,
        curriculum_init_noise_scale=post_cfg.reward.curriculum_init_noise_scale,
        max_steps=post_cfg.max_steps,
    )
    complexity_reward_fn = ComplexityRewardFn(
        tokenizer=tokenizer,
        job_config=job_config,
        simplipy_engine_name=post_cfg.reward.simplipy_engine_name,
        standardize_dataset=post_cfg.reward.standardize_dataset,
    )

    grpo_args = _GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=post_cfg.learning_rate,
        per_device_train_batch_size=post_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=post_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=post_cfg.gradient_accumulation_steps,
        max_steps=post_cfg.max_steps,
        num_generations=post_cfg.num_generations,
        num_generations_eval=post_cfg.num_generations_eval,
        steps_per_generation=post_cfg.steps_per_generation,
        beta=post_cfg.beta,
        temperature=post_cfg.sample_temperature,
        top_k=post_cfg.top_k,
        top_p=post_cfg.top_p,
        warmup_ratio=post_cfg.warmup_ratio,
        lr_scheduler_type=post_cfg.lr_scheduler_type,
        eval_strategy="steps",
        save_steps=post_cfg.save_steps,
        eval_steps=post_cfg.eval_steps,
        logging_steps=post_cfg.logging_steps,
        report_to=post_cfg.report_to,
        remove_unused_columns=False,
        seed=post_cfg.seed,
        reward_weights=[1.0, post_cfg.reward.complexity_weight],
        scale_rewards=post_cfg.scale_rewards,
        multi_objective_aggregation=post_cfg.multi_objective_aggregation,
        max_completion_length=max_completion_length,
        log_completions=True,
        num_train_epochs=1,
        weight_decay=post_cfg.weight_decay,
        adam_beta1=post_cfg.beta1,
        adam_beta2=post_cfg.beta2,
        bf16=True,
        gradient_checkpointing=False,
    )

    trainer = CustomGRPOTrainer(
        model=policy_wrapper,
        ref_model=ref_wrapper,
        reward_funcs=[nrmse_reward_fn, complexity_reward_fn],
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        model_config_path=str(model_config_path),
        rs_grpo_beta=post_cfg.rs_grpo_beta,
        eval_generation_chunk_size=post_cfg.eval_generation_chunk_size,
        train_generation_chunk_size=post_cfg.train_generation_chunk_size,
        offload_ref_model=post_cfg.offload_ref_model,
    )

    trainer.train()

    if not trainer.is_world_process_zero():
        return
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_weights(policy_wrapper.model, final_dir)
    shutil.copy2(model_config_path, final_dir / "config.toml")
    shutil.copy2(output_dir / "post_train.toml", final_dir / "post_train.toml")


if __name__ == "__main__":
    main()

