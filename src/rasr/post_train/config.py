from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DataloaderConfig:
    data_path: str
    set_name: str
    num_vars: int
    val_set_name: str | None = None


@dataclass(slots=True)
class RewardConfig:
    refine_constants: bool
    standardize_dataset: bool
    n_restarts: int
    init_noise_scale: float
    refinement_timeout: float
    complexity_weight: float
    simplipy_engine_name: str
    curriculum_init_noise_scale: bool


@dataclass(slots=True)
class PostTrainConfig:
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    num_generations: int
    num_generations_eval: int
    eval_generation_chunk_size: int
    train_generation_chunk_size: int
    steps_per_generation: int
    max_steps: int
    max_completion_length: int
    beta: float
    temperature: float
    top_k: int
    top_p: float
    sample_temperature: float
    warmup_ratio: float
    lr_scheduler_type: str
    save_steps: int
    eval_steps: int
    logging_steps: int
    report_to: list[str]
    seed: int
    scale_rewards: str
    multi_objective_aggregation: str
    reward: RewardConfig
    dataloader: DataloaderConfig
    rs_grpo_beta: float
    offload_ref_model: bool = False
    gradient_checkpointing: bool = False
    output_dir: str | None = None
    run_name: str | None = None

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "PostTrainConfig":
        path = Path(config_path)
        with path.open("rb") as f:
            payload = tomllib.load(f)
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: dict) -> "PostTrainConfig":
        section = payload["post_training"]
        if not isinstance(section, dict):
            raise ValueError("[post_training] must be a table.")

        dl_payload = section["dataloader"]
        if not isinstance(dl_payload, dict):
            raise ValueError("[post_training.dataloader] must be a table.")

        dataloader = DataloaderConfig(
            data_path=str(dl_payload["data_path"]),
            set_name=str(dl_payload["set_name"]),
            num_vars=int(dl_payload["num_vars"]),
            val_set_name=str(dl_payload["val_set_name"]) if dl_payload.get("val_set_name") else None,
        )

        reward_payload = section["reward"]
        if not isinstance(reward_payload, dict):
            raise ValueError("[post_training.reward] must be a table.")

        reward = RewardConfig(
            refine_constants=bool(reward_payload["refine_constants"]),
            standardize_dataset=bool(reward_payload.get("standardize_dataset", False)),
            n_restarts=int(reward_payload["n_restarts"]),
            init_noise_scale=float(reward_payload["init_noise_scale"]),
            refinement_timeout=float(reward_payload.get("refinement_timeout", 30.0)),
            complexity_weight=float(reward_payload["complexity_weight"]),
            simplipy_engine_name=str(reward_payload["simplipy_engine_name"]),
            curriculum_init_noise_scale=bool(reward_payload.get("curriculum_init_noise_scale", False)),
        )

        report_to_raw = section["report_to"]
        if isinstance(report_to_raw, str):
            report_to = [report_to_raw]
        elif isinstance(report_to_raw, list):
            report_to = [str(item) for item in report_to_raw]
        else:
            raise ValueError("post_training.report_to must be a string or a list of strings.")

        return cls(
            learning_rate=float(section["learning_rate"]),
            weight_decay=float(section["weight_decay"]),
            beta1=float(section["beta1"]),
            beta2=float(section["beta2"]),
            per_device_train_batch_size=int(section["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(section["gradient_accumulation_steps"]),
            per_device_eval_batch_size=int(section["per_device_eval_batch_size"]),
            num_generations=int(section["num_generations"]),
            num_generations_eval=int(section["num_generations_eval"]),
            eval_generation_chunk_size=int(section["eval_generation_chunk_size"]),
            train_generation_chunk_size=int(section["train_generation_chunk_size"]),
            steps_per_generation=int(section["steps_per_generation"]),
            max_steps=int(section["max_steps"]),
            max_completion_length=int(section["max_completion_length"]),
            beta=float(section["beta"]),
            temperature=float(section["temperature"]),
            top_k=int(section["top_k"]),
            top_p=float(section["top_p"]),
            sample_temperature=float(section["sample_temperature"]),
            warmup_ratio=float(section["warmup_ratio"]),
            lr_scheduler_type=str(section["lr_scheduler_type"]),
            save_steps=int(section["save_steps"]),
            eval_steps=int(section["eval_steps"]),
            logging_steps=int(section["logging_steps"]),
            output_dir=section.get("output_dir"),
            run_name=section.get("run_name"),
            report_to=report_to,
            seed=int(section["seed"]),
            scale_rewards=str(section["scale_rewards"]),
            multi_objective_aggregation=str(section["multi_objective_aggregation"]),
            rs_grpo_beta=float(section["rs_grpo_beta"]),
            offload_ref_model=bool(section.get("offload_ref_model", False)),
            gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
            reward=reward,
            dataloader=dataloader,
        )

