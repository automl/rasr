from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class ParameterLoggingConfig:
    """Configuration for parameter statistics logging."""
    enabled: bool = False
    log_interval: int = 100
    log_parameters: bool = True
    log_gradients: bool = True
    log_optimizer_states: bool = True
    log_activations: bool = False  # Activation norms after each layer's residual update.
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


class ParameterStatsLogger:
    """Compute parameter, gradient, optimizer-state and activation statistics for TensorBoard."""

    def __init__(
        self,
        config: Union[ParameterLoggingConfig, Any],
        model_parts: Optional[List[torch.nn.Module]] = None,
        optimizers: Optional[Any] = None,
    ):
        if hasattr(config, 'enabled'):
            self.config = ParameterLoggingConfig(
                enabled=config.enabled,
                log_interval=config.log_interval,
                log_parameters=config.log_parameters,
                log_gradients=config.log_gradients,
                log_optimizer_states=config.log_optimizer_states,
                log_activations=getattr(config, 'log_activations', False),
                include_patterns=config.include_patterns,
                exclude_patterns=config.exclude_patterns,
            )
        else:
            self.config = config

        self.model_parts = model_parts or []
        self.optimizers = optimizers
        self.last_log_step = -1

    def should_log(self, step: int) -> bool:
        """Check if we should log statistics at this step."""
        if not self.config.enabled:
            return False

        if step == 1:
            return True

        return step % self.config.log_interval == 0

    def _matches_patterns(self, name: str) -> bool:
        """Check if parameter name matches include/exclude patterns."""
        if self.config.exclude_patterns:
            for pattern in self.config.exclude_patterns:
                if pattern in name:
                    return False

        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if pattern in name:
                    return True
            return False

        return True

    def _iter_model_parameters(self):
        """Yield (model_prefix, name, param) for trainable, pattern-matched parameters."""
        for model_idx, model in enumerate(self.model_parts):
            model_prefix = f"model_{model_idx}" if len(self.model_parts) > 1 else "model"

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if not self._matches_patterns(name):
                    continue

                clean_name = name.replace('.', '/')
                yield model_prefix, clean_name, param

    def _get_tensor_stats(self, tensor: torch.Tensor, prefix: str) -> Dict[str, float]:
        """Compute statistics for a tensor, handling DTensor if present."""
        if tensor is None:
            return {}

        if hasattr(tensor, 'full_tensor'):
            tensor_data = tensor.full_tensor()
        else:
            tensor_data = tensor

        with torch.no_grad():
            try:
                if tensor_data.dim() == 1:
                    stats = {
                        # f"{prefix}/max": tensor_data.max().item(),
                        # f"{prefix}/min": tensor_data.min().item(),
                        f"{prefix}/mean": tensor_data.mean().item(),
                        f"{prefix}/std": tensor_data.std().item(),
                    }
                elif tensor_data.dim() >= 2:
                    stats = {
                    # f"{prefix}/max": tensor_data.max().item(),
                    # f"{prefix}/min": tensor_data.min().item(),
                    f"{prefix}/norm": tensor_data.norm().item(),
                    f"{prefix}/mean": tensor_data.mean().item(),
                    f"{prefix}/norm_md1": tensor_data.norm(dim=1).mean().item(),
                    f"{prefix}/std": tensor_data.std().item(),
                }
                else:
                    stats = {
                    f"{prefix}/scalar": tensor_data.item(),
                }
                return stats
            except Exception:
                # Numeric sentinel, since TensorBoard can't log strings as scalars.
                return {f"{prefix}/error": -1.0}

    def _compute_parameter_stats(self) -> Dict[str, Any]:
        """Compute parameter statistics for all model parts."""
        if not self.config.log_parameters:
            return {}

        param_stats = {}
        for model_prefix, clean_name, param in self._iter_model_parameters():
            prefix = f"params/{model_prefix}/{clean_name}"
            stats = self._get_tensor_stats(param.data, prefix)
            param_stats.update(stats)

        return param_stats

    def _compute_gradient_stats(self) -> Dict[str, Any]:
        """Compute gradient statistics for all model parts."""
        if not self.config.log_gradients:
            return {}

        grad_stats = {}
        for model_prefix, clean_name, param in self._iter_model_parameters():
            if param.grad is not None:
                prefix = f"grads/{model_prefix}/{clean_name}"
                stats = self._get_tensor_stats(param.grad.data, prefix)
                grad_stats.update(stats)

        return grad_stats

    def _compute_optimizer_stats(self) -> Dict[str, Any]:
        """Compute AdamW optimizer statistics."""
        if not self.config.log_optimizer_states or self.optimizers is None:
            return {}

        optimizer_stats = {}

        optimizers_list = (
            list(self.optimizers.optimizers) if hasattr(self.optimizers, 'optimizers')
            else [self.optimizers]
        )

        for opt_idx, optimizer in enumerate(optimizers_list):
            opt_prefix = f"optimizer_{opt_idx}" if len(optimizers_list) > 1 else "optimizer"

            if not hasattr(optimizer, 'state'):
                continue

            for group_idx, param_group in enumerate(optimizer.param_groups):
                for param_idx, param in enumerate(param_group['params']):
                    if param not in optimizer.state:
                        continue

                    state = optimizer.state[param]
                    param_prefix = f"{opt_prefix}/group_{group_idx}/param_{param_idx}"

                    if 'exp_avg' in state:
                        exp_avg_stats = self._get_tensor_stats(
                            state['exp_avg'],
                            f"{param_prefix}/exp_avg"
                        )
                        optimizer_stats.update(exp_avg_stats)

                    if 'exp_avg_sq' in state:
                        exp_avg_sq_stats = self._get_tensor_stats(
                            state['exp_avg_sq'],
                            f"{param_prefix}/exp_avg_sq"
                        )
                        optimizer_stats.update(exp_avg_sq_stats)

        return optimizer_stats


    def _compute_parameter_count(self) -> Dict[str, Any]:
        """Compute total and trainable parameter counts from model parts."""
        if not self.model_parts:
            return {}

        total_params = 0
        trainable_params = 0
        for model in self.model_parts:
            for param in model.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

        return {
            'parameter_count/total': float(total_params),
            'parameter_count/trainable': float(trainable_params),
        }

    def _compute_activation_stats(self) -> Dict[str, Any]:
        """Collect per-layer activation norms; requires set_activation_logging(True) on the model."""
        if not self.config.log_activations:
            return {}

        activation_stats = {}
        for model in self.model_parts:
            if hasattr(model, 'get_activation_norms'):
                norms = model.get_activation_norms()
                for layer_name, norm_value in norms.items():
                    activation_stats[f'activations/{layer_name}/mean_norm'] = (
                        norm_value.item() if isinstance(norm_value, torch.Tensor) else norm_value
                    )
        return activation_stats

    def enable_model_activation_logging(self) -> None:
        """Enable activation logging on all model parts that support it."""
        if not self.config.log_activations or not self.model_parts:
            return
        for model in self.model_parts:
            if hasattr(model, 'set_activation_logging'):
                model.set_activation_logging(True)

    def disable_model_activation_logging(self) -> None:
        """Disable activation logging on all model parts."""
        if not self.model_parts:
            return
        for model in self.model_parts:
            if hasattr(model, 'set_activation_logging'):
                model.set_activation_logging(False)

    def compute_stats(self, step: int) -> Dict[str, Any]:
        """Compute all enabled statistics as a flat metrics dict."""
        if not self.config.enabled:
            return {}

        all_stats = {}

        param_stats = self._compute_parameter_stats()
        all_stats.update(param_stats)

        grad_stats = self._compute_gradient_stats()
        all_stats.update(grad_stats)

        optimizer_stats = self._compute_optimizer_stats()
        all_stats.update(optimizer_stats)

        activation_stats = self._compute_activation_stats()
        all_stats.update(activation_stats)

        # Parameter counts don't change, so only log them once.
        if self.last_log_step == -1:
            param_count_stats = self._compute_parameter_count()
            all_stats.update(param_count_stats)

        self.last_log_step = step

        return all_stats

    def update_model_parts(self, model_parts: List[torch.nn.Module]) -> None:
        """Update model parts reference."""
        self.model_parts = model_parts

    def update_optimizers(self, optimizers: Any) -> None:
        """Update optimizers reference."""
        self.optimizers = optimizers