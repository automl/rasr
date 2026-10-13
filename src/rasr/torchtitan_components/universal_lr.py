"""Three-phase (warm, main, cooldown) learning rate scheduler."""

import math

from torch.optim.lr_scheduler import _LRScheduler


class UniversalLR(_LRScheduler):
    """Three-phase LR scheduler: warm-up/down, main decay, then cooldown to lr_min_absolute.

    Each phase duration is given by either `*_steps` or `*_ratio` (fraction of total_steps), not both.
    With warm_direction="down" the LR starts at warm_start_ratio * base_lr; main_decay_ratio is the
    fraction of base_lr reached at the end of the main phase.
    """

    def __init__(
        self,
        optimizer,
        total_steps,
        warm_steps=0,
        warm_ratio=0.0,
        warm_direction="up",
        warm_type="linear",
        warm_start_ratio=2.0,
        main_decay_type="const",
        main_decay_ratio=0.2,
        cooldown_steps=0,
        cooldown_ratio=0.0,
        cooldown_type="cosine",
        lr_min_absolute=0,
        last_epoch=-1,
    ):
        self.total_steps = total_steps
        self.lr_min_absolute = lr_min_absolute

        self.warm_steps = int(warm_steps)
        self.warm_ratio = float(warm_ratio)
        self.warm_direction = warm_direction
        self.warm_type = warm_type
        self.warm_start_ratio = warm_start_ratio

        self.main_decay_type = main_decay_type
        self.main_decay_ratio = main_decay_ratio

        self.cooldown_steps = int(cooldown_steps)
        self.cooldown_ratio = float(cooldown_ratio)
        self.cooldown_type = cooldown_type

        # Must run before super().__init__, which already calls get_lr().
        self._validate_and_compute_phases()

        super().__init__(optimizer, last_epoch)

        self._main_end_lrs = [None] * len(self.optimizer.param_groups)

    def _validate_and_compute_phases(self):
        """Validate parameters and compute phase durations."""
        if self.warm_steps > 0 and self.warm_ratio > 0:
            raise ValueError(
                f"warm_steps ({self.warm_steps}) and warm_ratio ({self.warm_ratio}) are mutually exclusive. "
                f"At most one can be non-zero."
            )

        if self.warm_ratio > 0:
            self.warm_steps = int(self.total_steps * self.warm_ratio)

        if self.warm_direction not in ["up", "down"]:
            raise ValueError(f"warm_direction must be 'up' or 'down', got '{self.warm_direction}'")

        if self.warm_type not in ["linear", "cosine", "exp", "sqrt"]:
            raise ValueError(
                f"warm_type must be 'linear', 'cosine', 'exp', or 'sqrt', got '{self.warm_type}'"
            )

        if self.cooldown_steps > 0 and self.cooldown_ratio > 0:
            raise ValueError(
                f"cooldown_steps ({self.cooldown_steps}) and cooldown_ratio ({self.cooldown_ratio}) "
                f"are mutually exclusive. "
                f"At most one can be non-zero."
            )

        if self.cooldown_ratio > 0:
            self.cooldown_steps = int(self.total_steps * self.cooldown_ratio)

        if self.cooldown_type not in ["linear", "cosine", "exp", "sqrt"]:
            raise ValueError(
                f"cooldown_type must be 'linear', 'cosine', 'exp', or 'sqrt', got '{self.cooldown_type}'"
            )

        if self.main_decay_type not in ["const", "linear", "cosine", "exp", "sqrt"]:
            raise ValueError(
                f"main_decay_type must be 'const', 'linear', 'cosine', 'exp', or 'sqrt', got '{self.main_decay_type}'"
            )

        self.main_steps = self.total_steps - self.warm_steps - self.cooldown_steps

        if self.main_steps < 0:
            raise ValueError(
                f"Warm phase + cooldown phase must be < total_steps. "
                f"Got warm={self.warm_steps}, cooldown={self.cooldown_steps}, total={self.total_steps}"
            )

    def _compute_decay_factor(self, progress, decay_type):
        """Map progress in [0, 1] to a decay factor going from 1.0 to 0.0."""
        progress = max(0.0, min(1.0, progress))

        if decay_type == "linear":
            return 1.0 - progress

        elif decay_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        elif decay_type == "sqrt":
            return math.sqrt(1.0 - progress)

        elif decay_type == "exp":
            # Not a true exponential; quadratic decay.
            return (1.0 - progress) ** 2

        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")

    def _compute_warm_lr(self, base_lr, t_warm):
        """Compute the warm phase learning rate."""
        if self.warm_steps == 0:
            return base_lr

        progress = t_warm / max(1, self.warm_steps - 1)

        if self.warm_direction == "up":
            lr_start = self.lr_min_absolute
            lr_end = base_lr

            decay_factor = self._compute_decay_factor(progress, self.warm_type)
            lr = lr_end + (lr_start - lr_end) * decay_factor

        else:  # warm_direction == "down"
            lr_start = self.warm_start_ratio * base_lr
            lr_end = base_lr

            decay_factor = self._compute_decay_factor(progress, self.warm_type)
            lr = lr_end + (lr_start - lr_end) * decay_factor

        return lr

    def _compute_main_base_lr(self, base_lr, t_main):
        """Compute the main phase learning rate."""
        if self.main_decay_type == "const":
            return base_lr

        progress = t_main / max(1, self.main_steps - 1)
        decay_factor = self._compute_decay_factor(progress, self.main_decay_type)

        # Floor at lr_min_absolute so main_decay_ratio=0.0 anneals to it.
        target_lr = max(self.lr_min_absolute, base_lr * self.main_decay_ratio)
        lr = target_lr + (base_lr - target_lr) * decay_factor

        return lr

    def _compute_cooldown_lr(self, lr_start, t_cooldown):
        """Compute the cooldown phase learning rate, decaying from lr_start to lr_min_absolute."""
        if self.cooldown_steps == 0:
            return lr_start

        progress = t_cooldown / max(1, self.cooldown_steps - 1)
        decay_factor = self._compute_decay_factor(progress, self.cooldown_type)

        lr = self.lr_min_absolute + (lr_start - self.lr_min_absolute) * decay_factor

        return lr

    def get_lr(self):
        """Compute learning rate for current step."""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "use `get_last_lr()`.", UserWarning)

        t = self.last_epoch
        new_lrs = []

        for i, base_lr in enumerate(self.base_lrs):
            if t < self.warm_steps:
                lr = self._compute_warm_lr(base_lr, t)

            elif t < self.warm_steps + self.main_steps:
                t_main = t - self.warm_steps

                base_schedule_lr = self._compute_main_base_lr(base_lr, t_main)

                lr = base_schedule_lr

                lr = max(self.lr_min_absolute, lr)

                # Remember the final main-phase LR so cooldown starts from it.
                if t == self.warm_steps + self.main_steps - 1:
                    self._main_end_lrs[i] = lr

            else:
                t_cooldown = t - self.warm_steps - self.main_steps

                if self._main_end_lrs[i] is None:
                    lr_start = base_lr
                else:
                    lr_start = self._main_end_lrs[i]

                lr = self._compute_cooldown_lr(lr_start, t_cooldown)

            new_lrs.append(lr)

        return new_lrs
