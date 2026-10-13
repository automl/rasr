from dataclasses import dataclass, field

from torchtitan.config import JobConfig as BaseJobConfig
from torchtitan.config import LRScheduler as BaseLRScheduler
from torchtitan.config.job_config import (
    DataLoader as BaseDataLoader,
)
from torchtitan.config.job_config import (
    Training as BaseTraining,
)


@dataclass
class DataLoader(BaseDataLoader):
    """Extended dataloader config with RASR dataset location options."""

    data_path: str = ""
    """Path to directory containing {set_name}_dataset*.arrow files for RASR."""

    set_name: str = "train"
    """Dataset split name used as the glob prefix (train/val)."""

    val_set_name: str = "val"
    """Validation split prefix (e.g. val -> val_dataset*.arrow)."""


@dataclass
class Validation:
    """Validation configuration for evaluation during training."""

    enable: bool = False
    """Enable validation during training."""

    freq: int = 100
    """Validation frequency in training steps."""

    steps: int = -1
    """Number of validation steps (-1 = full validation set)."""

    local_batch_size: int = -1
    """Per-device validation batch size (-1 = training.local_batch_size)."""


@dataclass
class Tokenizer:
    """Equation tokenizer config (TOML [tokenizer])."""

    special_tokens: list[str] = field(default_factory=lambda: ["<SOE>", "<EOE>", "<PAD>"])
    operators: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    digits: list[str] = field(default_factory=lambda: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    special_chars: list[str] = field(default_factory=lambda: ["."])
    constant_strategy: str = "mantissa_exponent"

    # Filled by the RASR dataloader at runtime; don't set in TOML.
    trg_vocab_size: int = -1
    soe_token_id: int = -1
    eoe_token_id: int = -1
    pad_token_id: int = -1


@dataclass
class ParameterLogging:
    """Parameter statistics logging configuration."""

    enabled: bool = False
    """Enable parameter statistics logging to TensorBoard."""

    log_interval: int = 100
    """Logging interval in steps."""

    log_parameters: bool = True
    """Log parameter statistics."""

    log_gradients: bool = True
    """Log gradient statistics."""

    log_optimizer_states: bool = True
    """Log AdamW optimizer state statistics."""

    include_patterns: list[str] = field(default_factory=list)
    """Include only parameters matching these patterns (empty = all)."""

    exclude_patterns: list[str] = field(default_factory=list)
    """Exclude parameters matching these patterns."""


@dataclass
class LRScheduler(BaseLRScheduler):
    """LR scheduler config for the universal 3-phase scheduler (warm -> main -> cooldown)."""

    scheduler_type: str = "wsd"
    """Scheduler type; only "universal" is supported."""

    lr_min_absolute: float | None = None
    """Absolute minimum learning rate."""

    warm_steps: int = 0
    """Number of warm steps (mutually exclusive with warm_ratio)."""

    warm_ratio: float = 0.0
    """Warm steps as a ratio of total steps (mutually exclusive with warm_steps)."""

    warm_direction: str = "up"
    """Warm direction: "up" (lr_min_absolute -> base_lr) or "down" (warm_start_ratio * base_lr -> base_lr)."""

    warm_type: str = "linear"
    """Warm phase curve: "linear", "cosine", "sqrt" or "exp"."""

    warm_start_ratio: float = 2.0
    """Starting LR multiplier for warmdown (warm_direction="down" only)."""

    main_decay_type: str = "const"
    """Main phase curve: "const", or "linear"/"cosine"/"exp"/"sqrt" decay to main_decay_ratio * base_lr."""

    main_decay_ratio: float = 0.2
    """Target LR ratio at the end of the main phase (ignored when main_decay_type="const")."""

    cooldown_steps: int = 0
    """Number of cooldown steps (mutually exclusive with cooldown_ratio)."""

    cooldown_ratio: float = 0.0
    """Cooldown steps as a ratio of total steps (mutually exclusive with cooldown_steps)."""

    cooldown_type: str = "cosine"


@dataclass
class JobConfig(BaseJobConfig):
    """Torchtitan JobConfig extended with RASR-specific fields."""

    training: BaseTraining = field(default_factory=BaseTraining)
    dataloader: DataLoader = field(default_factory=DataLoader)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    validation: Validation = field(default_factory=Validation)
    tokenizer: Tokenizer = field(default_factory=Tokenizer)
    parameter_logging: ParameterLogging = field(default_factory=ParameterLogging)
