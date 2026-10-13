"""TorchTitan container and builders for the three-phase UniversalLR scheduler."""

import copy
from typing import Any, Dict, List

from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import LRScheduler as LRSchedulerConfig
from torchtitan.tools.logging import logger

from src.rasr.torchtitan_components.universal_lr import UniversalLR

__all__ = [
    "UniversalLRSchedulersContainer",
    "build_lr_schedulers_universal",
    "build_lr_schedulers_auto",
]


class UniversalLRSchedulersContainer(Stateful):
    """Container holding one UniversalLR scheduler per optimizer."""

    def __init__(
        self,
        optimizers: OptimizersContainer,
        lr_scheduler_config: LRSchedulerConfig,
        training_steps: int
    ) -> None:
        assert len(optimizers) > 0, "Must have at least one optimizer to create LRScheduler"

        warm_steps = getattr(lr_scheduler_config, 'warm_steps', 0)
        warm_ratio = getattr(lr_scheduler_config, 'warm_ratio', 0.0)
        warm_direction = getattr(lr_scheduler_config, 'warm_direction', 'up')
        warm_type = getattr(lr_scheduler_config, 'warm_type', 'linear')
        warm_start_ratio = getattr(lr_scheduler_config, 'warm_start_ratio', 2.0)

        main_decay_type = getattr(lr_scheduler_config, 'main_decay_type', 'const')
        main_decay_ratio = getattr(lr_scheduler_config, 'main_decay_ratio', 0.2)

        cooldown_steps = getattr(lr_scheduler_config, 'cooldown_steps', 0)
        cooldown_ratio = getattr(lr_scheduler_config, 'cooldown_ratio', 0.0)
        cooldown_type = getattr(lr_scheduler_config, 'cooldown_type', 'cosine')

        lr_min_absolute = getattr(lr_scheduler_config, 'lr_min_absolute', None)
        if lr_min_absolute is None:
            base_lr = optimizers.optimizers[0].param_groups[0]['lr']
            min_lr_factor = getattr(lr_scheduler_config, 'min_lr_factor', 0.0)
            lr_min_absolute = base_lr * min_lr_factor

        actual_warm_steps = warm_steps if warm_steps > 0 else int(training_steps * warm_ratio)
        actual_cooldown_steps = cooldown_steps if cooldown_steps > 0 else int(training_steps * cooldown_ratio)
        actual_main_steps = training_steps - actual_warm_steps - actual_cooldown_steps

        warm_desc = "disabled"
        if actual_warm_steps > 0:
            if warm_ratio > 0:
                warm_desc = f"{actual_warm_steps} steps ({warm_ratio*100:.1f}% of total)"
            else:
                warm_desc = f"{actual_warm_steps} steps (absolute)"
            warm_desc += f", direction={warm_direction}, type={warm_type}"
            if warm_direction == "down":
                warm_desc += f", start_ratio={warm_start_ratio}"

        cooldown_desc = "disabled"
        if actual_cooldown_steps > 0:
            if cooldown_ratio > 0:
                cooldown_desc = f"{actual_cooldown_steps} steps ({cooldown_ratio*100:.1f}% of total)"
            else:
                cooldown_desc = f"{actual_cooldown_steps} steps (absolute)"
            cooldown_desc += f", type={cooldown_type}"

        main_desc = f"{actual_main_steps} steps, decay_type={main_decay_type}, target_ratio={main_decay_ratio}"

        logger.info(
            f"Creating Universal LR Scheduler:\n"
            f"  Total steps: {training_steps}\n"
            f"  Phase distribution: {actual_warm_steps} warm → {actual_main_steps} main → "
            f"{actual_cooldown_steps} cooldown\n"
            f"  Phase 1 Warm: {warm_desc}\n"
            f"  Phase 2 Main: {main_desc}\n"
            f"  Phase 3 Cooldown: {cooldown_desc}\n"
            f"  Min LR absolute: {lr_min_absolute:.2e}\n"
        )

        self.schedulers: List[UniversalLR] = []
        for i, optimizer in enumerate(optimizers):
            scheduler = UniversalLR(
                optimizer=optimizer,
                total_steps=training_steps,
                warm_steps=warm_steps,
                warm_ratio=warm_ratio,
                warm_direction=warm_direction,
                warm_type=warm_type,
                warm_start_ratio=warm_start_ratio,
                main_decay_type=main_decay_type,
                main_decay_ratio=main_decay_ratio,
                cooldown_steps=cooldown_steps,
                cooldown_ratio=cooldown_ratio,
                cooldown_type=cooldown_type,
                lr_min_absolute=lr_min_absolute,
                last_epoch=-1,
            )
            self.schedulers.append(scheduler)

            initial_lrs = scheduler.get_last_lr()
            logger.info(f"Optimizer {i} initial LRs: {initial_lrs}")

    def __iter__(self):
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        """Step all schedulers and log current learning rates."""
        for i, scheduler in enumerate(self.schedulers):
            scheduler.step()

            if scheduler.last_epoch % 100 == 0:
                current_lrs = scheduler.get_last_lr()
                phase = self._get_current_phase(scheduler)
                logger.debug(
                    f"Step {scheduler.last_epoch} - Optimizer {i} LRs: {current_lrs} "
                    f"(Phase: {phase})"
                )

    def _get_current_phase(self, scheduler: UniversalLR) -> str:
        """Determine current phase of the scheduler."""
        current_step = scheduler.last_epoch

        if current_step < scheduler.warm_steps:
            return f"warm ({scheduler.warm_direction})"
        elif current_step < scheduler.warm_steps + scheduler.main_steps:
            return "main"
        else:
            return "cooldown"

    def state_dict(self) -> Dict[str, Any]:
        """Save the state of every scheduler."""
        state = {
            'num_schedulers': len(self.schedulers),
            'scheduler_states': []
        }

        for i, scheduler in enumerate(self.schedulers):
            scheduler_state = scheduler.state_dict()
            scheduler_state['scheduler_index'] = i
            state['scheduler_states'].append(scheduler_state)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of every scheduler."""
        num_schedulers = state_dict.get('num_schedulers', 1)
        if num_schedulers != len(self.schedulers):
            logger.warning(
                f"Checkpoint has {num_schedulers} schedulers but current config has "
                f"{len(self.schedulers)}. Will load available states."
            )

        scheduler_states = state_dict.get('scheduler_states', [])

        for i, scheduler in enumerate(self.schedulers):
            if i < len(scheduler_states):
                scheduler_state = copy.deepcopy(scheduler_states[i])
                scheduler_state.pop('scheduler_index', None)
                scheduler.load_state_dict(scheduler_state)
                logger.info(f"Loaded state for Universal scheduler {i} at step {scheduler.last_epoch}")
            else:
                logger.warning(f"No checkpoint state available for Universal scheduler {i}")


def build_lr_schedulers_universal(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> UniversalLRSchedulersContainer:
    """Build a UniversalLRSchedulersContainer for the given optimizers."""
    return UniversalLRSchedulersContainer(optimizers, lr_scheduler_config, training_steps)


def build_lr_schedulers_auto(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> UniversalLRSchedulersContainer:
    """Build universal LR schedulers, warning if a legacy scheduler_type is configured."""
    scheduler_type = getattr(lr_scheduler_config, 'scheduler_type', 'universal')

    if scheduler_type != "universal":
        logger.warning(
            f"scheduler_type='{scheduler_type}' is no longer supported. "
            f"All scheduler types have been unified into 'universal'. "
            f"Using universal scheduler instead.\n"
            f"Please update your config to use scheduler_type='universal' and configure "
            f"the three-phase architecture (warm/main/cooldown) to achieve the desired behavior.\n"
            f"See documentation for examples of how to emulate old scheduler types."
        )

    logger.info("Using Universal LR Scheduler (unified scheduler for all use cases)")
    return build_lr_schedulers_universal(optimizers, lr_scheduler_config, training_steps)
