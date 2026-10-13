from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.rasr.data.generation.constant_sampler import BottomUpConstantSampler
from src.rasr.data.generation.tabicl_v2_prior._activations import (
    AbsActivation,
    ExpActivation,
    RandomScaleLayer,
    SqrtAbsActivation,
    SquareActivation,
    StdScaleLayer,
)
from src.rasr.data.generation.tabicl_v2_prior._dataset import GraphPrior
from src.rasr.data.generation.tabicl_v2_prior.graph_lib._config import PriorConfig


class NoValidDatasetSampleFoundError(Exception):
    pass


class ConstantScaleLayer(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


def _build_num_samples_sampler(config: dict):
    distribution = config["distribution"]
    if distribution == "uniform_int":
        low = int(config["min"])
        high = int(config["max"])
        return lambda: int(round(np.random.uniform(low, high)))
    raise ValueError("num_samples only supports 'uniform_int'.")


class DatasetSampler:
    def __init__(
        self,
        config: Any,
        tokenizer_config: Any,
        operator_arity: dict[str, int],
        constant_sampling_config: dict,
        target_noise_scale_std: float,
        target_noise_scale_zero_ratio: float,
        num_extra_columns_std: float,
        backbone: str = "tabicl_v2",
        noise_config: dict | None = None,
        retries: int = 0,
        constant_max_exponent: int = 4,
        standardize_x: bool = True,
        standardize_y: bool = True,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu:0",
    ):
        self.num_samples_sampler = _build_num_samples_sampler(config["num_samples"])
        self.tokenizer_variables = list(tokenizer_config["variables"])
        self.num_prior_nodes = len(tokenizer_config["variables"]) + 1
        self.operator_arity = operator_arity
        self.target_noise_scale_std = target_noise_scale_std
        self.target_noise_scale_zero_ratio = target_noise_scale_zero_ratio
        if not 0.0 <= self.target_noise_scale_zero_ratio <= 1.0:
            raise ValueError("target_noise_scale_zero_ratio must be within [0, 1].")
        self.num_extra_columns_std = num_extra_columns_std
        if noise_config is None:
            raise ValueError("noise_config is required. Set [prior.datasets.noise] in the config file.")
        self.noise_cfg = noise_config
        self._backbone = backbone
        self.noise_activation_factories = (
            nn.Identity,
            nn.LeakyReLU,
            nn.ELU,
            nn.SELU,
            nn.SiLU,
            nn.ReLU,
            nn.Softplus,
            nn.Sigmoid,
            nn.ReLU6,
            ExpActivation,
            SqrtAbsActivation,
            SquareActivation,
            AbsActivation,
        )
        self.retries = int(retries)
        self._constant_max_abs = 10.0 ** int(constant_max_exponent)
        self._standardize_x = bool(standardize_x)
        self._standardize_y = bool(standardize_y)
        self.device = device
        if self._backbone == "tabicl_v2":
            self._graph_prior_kwargs = dict(
                regression=True,
                batch_size=1,
                batch_size_per_gp=1,
                min_features=self.num_prior_nodes,
                max_features=self.num_prior_nodes,
                min_seq_len=None,
                min_train_size=1,
                max_train_size=2,
                config=PriorConfig(fct_types='mlp,mlp,mlp,mlp,lin,lin,lin,lin,quad,quad,quad,quad,gp,gp,gp,gp,prod,prod,prod,prod,tree,disc'),
                n_jobs=1,
                numerical_only=True,
                device="cpu",
            )
        elif self._backbone != "uniform_random_range":
            raise ValueError(f"Unknown backbone: {self._backbone!r}. Expected 'tabicl_v2' or 'uniform_random_range'.")
        self.constant_sampler = BottomUpConstantSampler(constant_sampling_config)

    @classmethod
    def from_config(cls, config: dict) -> "DatasetSampler":
        datasets_cfg = config["prior"]["datasets"]
        tokenizer_cfg = config["tokenizer"]
        expression_cfg = config["prior"]["expressions"]
        retries = config["data_generation"]["sampling_retries"]["datasets"]
        constant_max_exponent = int(expression_cfg.get("constants", {}).get("constant_max_exponent", 4))
        return cls(
            config=datasets_cfg,
            tokenizer_config=tokenizer_cfg,
            operator_arity=expression_cfg["operator_arity"],
            constant_sampling_config=expression_cfg["constant_sampling"],
            target_noise_scale_std=float(datasets_cfg["target_noise_scale_std"]),
            target_noise_scale_zero_ratio=float(datasets_cfg["target_noise_scale_zero_ratio"]),
            num_extra_columns_std=float(datasets_cfg["num_extra_columns_std"]),
            backbone=str(datasets_cfg.get("backbone", "tabicl_v2")),
            noise_config=datasets_cfg["noise"],
            retries=int(retries),
            constant_max_exponent=constant_max_exponent,
            standardize_x=bool(datasets_cfg.get("standardize_x", True)),
            standardize_y=bool(datasets_cfg.get("standardize_y", True)),
        )

    def _resolve_num_samples(self, num_samples: int | None) -> int:
        if num_samples is not None:
            return int(num_samples)
        if self.num_samples_sampler is None:
            raise ValueError("No num_samples provided and no distribution configured.")
        return int(self.num_samples_sampler())

    def _sample_with_tabicl(self, num_samples: int) -> np.ndarray:
        if self._backbone == "uniform_random_range":
            return self._sample_uniform_random_range(num_samples)
        prior = GraphPrior(max_seq_len=num_samples, **self._graph_prior_kwargs)
        x_batch, _, d_batch, _, _ = prior.get_batch()
        x = x_batch[0]  # (num_samples, max_features)
        d = int(d_batch[0].item())
        x = x[:num_samples, :d]
        return x.detach().cpu().numpy()

    def _sample_uniform_random_range(self, num_samples: int) -> np.ndarray:
        """Sample each column uniformly on a random interval, following FLASH-ANSR (Appendix J.4)."""
        d = self.num_prior_nodes
        endpoints = np.clip(np.random.normal(0.0, 10.0, size=(d, 2)), -30.0, 30.0)
        lo = np.minimum(endpoints[:, 0], endpoints[:, 1])
        hi = np.maximum(endpoints[:, 0], endpoints[:, 1])
        hi = np.where(hi - lo < 1e-6, lo + 1e-6, hi)
        return np.random.uniform(lo, hi, size=(num_samples, d)).astype(np.float32)

    def _standardize_columns(self, arr: np.ndarray) -> np.ndarray:
        """Standardize each column to zero mean and unit variance."""
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        return (arr - mean) / std

    def _build_expression_variable_mapping(
        self,
        variables: list[str],
        permutation: np.ndarray | None,
    ) -> dict[str, str]:
        if permutation is None:
            return {v: v for v in variables}
        num_columns = int(permutation.shape[0])
        inverse_permutation = np.empty(num_columns, dtype=int)
        inverse_permutation[permutation] = np.arange(num_columns)
        variable_mapping: dict[str, str] = {}
        for original_idx, variable in enumerate(variables):
            new_idx = int(inverse_permutation[original_idx])
            variable_mapping[variable] = self.tokenizer_variables[new_idx]
        return variable_mapping

    def _sample_noise_activation(self) -> nn.Module:
        activation_idx = np.random.randint(len(self.noise_activation_factories))
        activation = self.noise_activation_factories[activation_idx]()
        flip_x = np.random.rand() < 0.5
        flip_y = np.random.rand() < 0.5
        return nn.Sequential(
            StdScaleLayer(),
            RandomScaleLayer(individual=False),
            ConstantScaleLayer(-1.0 if flip_x else 1.0),
            activation,
            ConstantScaleLayer(-1.0 if flip_y else 1.0),
        )

    def _sample_noise_plan(self, target_noise_scale: float | None) -> tuple[bool, float]:
        """Return whether base noise is active and its scale; `target_noise_scale_zero_ratio` is the clean rate."""
        std = self.target_noise_scale_std
        if std <= 0:
            raise ValueError("target_noise_scale_std must be positive.")
        base_prob = float(self.noise_cfg["base_noise_prob"])

        if target_noise_scale is not None:
            base_scale = float(target_noise_scale)
            return base_scale > 0.0, base_scale

        if np.random.rand() < self.target_noise_scale_zero_ratio:
            return False, 0.0

        base_present = np.random.rand() < base_prob
        base_scale = float(np.random.exponential(scale=std)) if base_present else 0.0
        return base_present, base_scale

    def _sample_num_extra_columns(self, max_extra_columns: int, num_extra_columns: int | None = None) -> int:
        if num_extra_columns is not None:
            return min(num_extra_columns, max_extra_columns)
        std = float(self.num_extra_columns_std)
        if std <= 0.0:
            return 0
        sampled = int(np.rint(stats.truncnorm.rvs(0.0, np.inf, loc=0.0, scale=std, size=1)[0]))
        return min(sampled, max_extra_columns)

    @staticmethod
    def _weighted_choice(weights: dict) -> str:
        """Sample a key from a {name: weight} dict, proportional to (non-negative) weights."""
        keys = list(weights.keys())
        w = np.array([max(0.0, float(weights[k])) for k in keys], dtype=np.float64)
        total = float(w.sum())
        if total <= 0.0:
            return keys[0]
        return keys[int(np.random.choice(len(keys), p=w / total))]

    def _activated_prior_noise(self, prior_node: np.ndarray) -> np.ndarray:
        """Pass a prior node through a random activation sandwich."""
        noise_input = torch.as_tensor(prior_node, dtype=torch.float32, device=self.device).reshape(-1, 1)
        activation = self._sample_noise_activation()
        with torch.no_grad():
            out = activation(noise_input).reshape(-1)
        return out.detach().cpu().numpy()

    def _sample_base_noise(self, n: int, prior_node: np.ndarray) -> tuple[np.ndarray, str]:
        """Draw unit-variance base noise `z` and the name of its distribution."""
        dist = self._weighted_choice(self.noise_cfg["distribution_weights"])
        if dist == "prior_node":
            raw = np.asarray(self._activated_prior_noise(prior_node), dtype=np.float64).reshape(-1)
            raw = raw - raw.mean()
            std = float(raw.std())
            if np.isfinite(std) and std >= 1e-8:
                return (raw / std).astype(np.float32), dist
            dist = "gaussian"  # Degenerate node, fall back to a Gaussian.
        raw = np.random.randn(n)
        z = (raw - raw.mean()) / (float(raw.std()) + 1e-8)
        return z.astype(np.float32), dist

    def _sample_variance_profile(self, scale_node: np.ndarray) -> tuple[np.ndarray, str]:
        """Return a per-row noise scale with mean 1. Data dependence lives in the variance so E[y|x] = f(x)."""
        n = scale_node.shape[0]
        mode = self._weighted_choice(self.noise_cfg["variance_weights"])
        if mode == "homoscedastic":
            return np.ones(n, dtype=np.float32), mode
        cfg = self.noise_cfg["hetero"]
        s = np.logaddexp(0.0, np.asarray(scale_node, dtype=np.float64)).reshape(-1)
        mean_s = float(s.mean())
        if not np.isfinite(mean_s) or mean_s < 1e-8:
            return np.ones(n, dtype=np.float32), mode
        strength = float(np.random.uniform(0.0, cfg["max_strength"]))
        sigma = (1.0 - strength) + strength * (s / mean_s)
        max_ratio = float(cfg["max_sigma_ratio"])
        sigma = np.clip(sigma, 0.0, max_ratio)
        sigma_mean = float(sigma.mean())
        if sigma_mean < 1e-8:
            return np.ones(n, dtype=np.float32), mode
        sigma = sigma / sigma_mean
        sigma = np.minimum(sigma, max_ratio)
        return sigma.astype(np.float32), mode

    def _validate_output(self, y: np.ndarray) -> bool:
        """Check validity of the raw (pre-standardization) y."""
        if len(y) == 0:
            return False
        var_y = float(np.var(y))
        if var_y < 1e-6:
            return False
        if float(np.max(np.abs(y))) > 1e6:
            return False
        # Reject heavy tails where the top 1% of points contribute over 90% of the variance.
        sq = (y - y.mean()) ** 2
        top_n = max(1, int(0.01 * len(y)))
        sorted_sq = np.sort(sq)[::-1]
        total = float(sorted_sq.sum())
        if total > 0 and float(sorted_sq[:top_n].sum()) / total > 0.9:
            return False
        return True

    def sample(
        self,
        skeleton: tuple[str, ...],
        variables: list[str],
        num_samples: int | None = None,
        target_noise_scale: float | None = None,
        num_extra_columns: int | None = None,
        filter_iqr: bool = True,
    ) -> tuple | None:
        """Sample a (X̃, ŷ, variable_mapping, noise_scale, constants, μ_y, σ_y) tuple."""
        target_num_samples = self._resolve_num_samples(num_samples)
        tolerance_factor = 0.25 if filter_iqr else 0.20
        samples_tolerance = int(target_num_samples * tolerance_factor)
        num_samples_plus = target_num_samples + samples_tolerance

        num_features = len(variables)

        sampled_nodes = self._sample_with_tabicl(num_samples_plus)
        if sampled_nodes.shape[1] < num_features + 1:
            return None

        if self._standardize_x:
            sampled_nodes = self._standardize_columns(sampled_nodes)

        real_features = sampled_nodes[:, :num_features]
        potential_correlated = sampled_nodes[:, num_features:-1]
        noise_node = sampled_nodes[:, -1]

        if potential_correlated.shape[1] > 0:
            n_extra = self._sample_num_extra_columns(
                max_extra_columns=potential_correlated.shape[1],
                num_extra_columns=num_extra_columns,
            )
            if n_extra > 0:
                x_data = np.concatenate([real_features, potential_correlated[:, :n_extra]], axis=1)
            else:
                x_data = real_features
        else:
            x_data = real_features

        real_features = x_data[:, :num_features]

        raw_constants, y_clean_raw = self.constant_sampler.sample_and_evaluate(
            skeleton, real_features, variables, self.operator_arity
        )
        if any(abs(c) >= self._constant_max_abs or abs(c) <= 1.0 / self._constant_max_abs for c in raw_constants):
            return None
        if np.isscalar(y_clean_raw):
            y_clean = np.full(real_features.shape[0], float(y_clean_raw), dtype=np.float32)
        else:
            y_clean = np.asarray(y_clean_raw, dtype=np.float32).reshape(-1)

        base_present, sampled_noise_scale = self._sample_noise_plan(target_noise_scale)

        # The prior_node shape is already feature-correlated, so it skips the variance profile.
        y_std = float(np.std(y_clean))
        if base_present and sampled_noise_scale > 0.0 and y_std > 0.0:
            z, z_dist = self._sample_base_noise(y_clean.shape[0], noise_node)
            if z_dist == "prior_node":
                sigma = np.ones(y_clean.shape[0], dtype=np.float32)
            else:
                sigma, _ = self._sample_variance_profile(noise_node)
            y_data = (y_clean + sampled_noise_scale * sigma * z * y_std).astype(np.float32)
        else:
            y_data = y_clean.astype(np.float32)

        valid_y = np.isfinite(y_data)
        valid_x = np.all(np.isfinite(x_data), axis=1)
        valid_mask = valid_y & valid_x
        invalid_count = int(valid_mask.shape[0] - np.sum(valid_mask))
        if invalid_count > samples_tolerance:
            return None

        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]

        if filter_iqr and len(y_data) > 0:
            q1, q3 = np.percentile(y_data, [25, 75])
            iqr_val = float(q3 - q1)
            iqr_mask = (y_data >= q1 - 5.0 * iqr_val) & (y_data <= q3 + 5.0 * iqr_val)
            x_data = x_data[iqr_mask]
            y_data = y_data[iqr_mask]

        if len(y_data) < target_num_samples:
            return None

        x_data = x_data[:target_num_samples]
        y_data = y_data[:target_num_samples]

        if not self._validate_output(y_data):
            return None

        mu_y_actual = float(np.mean(y_data))
        sigma_y_actual = float(np.std(y_data))
        if sigma_y_actual < 1e-6 or sigma_y_actual > 1e3:
            return None
        if self._standardize_y:
            mu_y, sigma_y = mu_y_actual, sigma_y_actual
            y_hat = ((y_data - mu_y) / sigma_y).astype(np.float32)
        else:
            mu_y, sigma_y = 0.0, 1.0
            y_hat = y_data.astype(np.float32)

        permutation: np.ndarray | None = None
        if x_data.shape[1] > 1:
            permutation = np.random.permutation(x_data.shape[1])
            x_data = x_data[:, permutation]

        variable_mapping = self._build_expression_variable_mapping(variables, permutation)

        return x_data, y_hat, variable_mapping, sampled_noise_scale, raw_constants, mu_y, sigma_y

    def sample_with_retries(
        self,
        skeleton: tuple[str, ...],
        variables: list[str],
        num_samples: int | None = None,
        target_noise_scale: float | None = None,
        num_extra_columns: int | None = None,
        retries: int | None = None,
        filter_iqr: bool = True,
    ) -> tuple | None:
        max_retries = self.retries if retries is None else int(retries)
        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                result = self.sample(
                    skeleton=skeleton,
                    variables=variables,
                    num_samples=num_samples,
                    target_noise_scale=target_noise_scale,
                    num_extra_columns=num_extra_columns,
                    filter_iqr=filter_iqr,
                )
                if result is not None:
                    return result
                last_error = NoValidDatasetSampleFoundError("Sample contains too many invalid points.")
            except Exception as error:
                last_error = error
                continue
        raise NoValidDatasetSampleFoundError("Unable to sample a valid dataset.") from last_error
