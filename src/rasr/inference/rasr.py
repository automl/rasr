"""Modified version of the sklearn-style estimator in https://github.com/psaegert/flash-ansr/tree/main"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import warnings
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import simplipy as sp
import torch
from huggingface_hub import snapshot_download
from loky import ProcessPoolExecutor
from loky.process_executor import TerminatedWorkerError
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

from src.rasr.data.tokenization import EquationTokenizer
from src.rasr.inference.checkpoint import load_rasr_model
from src.rasr.inference.expression_parser import (
    expression_to_lambda,
    is_numeric_string,
    parse_prefix_tokens,
    required_variable_count,
)
from src.rasr.inference.metrics import r2_score
from src.rasr.inference.refiner import ConvergenceError, Refiner
from src.rasr.inference.standardization import remap_to_original_space
from src.rasr.model.model import RASRTransformer


@dataclass
class CandidateResult:
    refiner: Refiner
    tokens: list[str]
    expression_string: str
    score: float
    r2: float
    log_prob: float
    y_pred: np.ndarray
    lambda_function: Callable[[np.ndarray], np.ndarray]
    constants: list[float]
    complexity: int = 0
    val_r2: float = float("nan")


def _pareto_front(results: list[CandidateResult], use_val: bool = False) -> list[CandidateResult]:
    """Return the non-dominated (fit, complexity) candidates, sorted by complexity."""
    if not results:
        return []
    if use_val:
        objectives = [(-r.val_r2, float(r.complexity)) for r in results]
    else:
        objectives = [(-r.r2, float(r.complexity)) for r in results]
    n = len(results)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if all(objectives[j][k] <= objectives[i][k] for k in range(2)) and any(
                objectives[j][k] < objectives[i][k] for k in range(2)
            ):
                dominated[i] = True
                break
    front_idx = [i for i in range(n) if not dominated[i]]
    seen_objectives: set[tuple[float, float]] = set()
    deduped_idx = []
    for i in front_idx:
        if objectives[i] in seen_objectives:
            continue
        seen_objectives.add(objectives[i])
        deduped_idx.append(i)
    front = [results[i] for i in deduped_idx]
    front.sort(key=lambda r: r.complexity)
    return front


_GLOBAL_ENGINE: sp.SimpliPyEngine | None = None
_GLOBAL_X: np.ndarray | None = None
_GLOBAL_Y: np.ndarray | None = None
_GLOBAL_N_VARIABLES: int | None = None
_WORKER_THREAD_LIMITER: Any = None  # kept alive so the BLAS limit persists


def _limit_worker_threads() -> None:
    """Limit a pool worker to single-thread BLAS to avoid oversubscription (never call in the parent)."""
    global _WORKER_THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits
        _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:
        pass


def _init_refinement_worker(
    simplipy_engine_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_variables: int,
) -> None:
    global _GLOBAL_ENGINE, _GLOBAL_X, _GLOBAL_Y, _GLOBAL_N_VARIABLES
    _GLOBAL_ENGINE = sp.SimpliPyEngine.load(simplipy_engine_name, install=True)
    _GLOBAL_X = X
    _GLOBAL_Y = y
    _GLOBAL_N_VARIABLES = n_variables


def _init_pooled_refinement_worker(
    simplipy_engine_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_variables: int,
) -> None:
    """Initializer for the per-call fork pool."""
    _limit_worker_threads()
    _init_refinement_worker(simplipy_engine_name, X, y, n_variables)


def _init_persistent_refinement_worker(simplipy_engine_name: str) -> None:
    """Initializer for the persistent pool; X/y are passed with each job instead."""
    global _GLOBAL_ENGINE
    _limit_worker_threads()
    _GLOBAL_ENGINE = sp.SimpliPyEngine.load(simplipy_engine_name, install=True)


def _resolve_refinement_arrays(payload: dict[str, Any]) -> tuple[sp.SimpliPyEngine, np.ndarray, np.ndarray, int]:
    """Take X/y from the job payload (persistent pool) or from the worker globals."""
    engine = _GLOBAL_ENGINE
    X = payload.get("X")
    y = payload.get("y")
    n_variables = payload.get("n_variables")
    if X is None or y is None or n_variables is None:
        X, y, n_variables = _GLOBAL_X, _GLOBAL_Y, _GLOBAL_N_VARIABLES
    if engine is None or X is None or y is None or n_variables is None:
        raise RuntimeError("Refinement worker context not initialized.")
    return engine, X, y, n_variables


def _refine_candidate_worker(payload: dict[str, Any]) -> dict[str, Any] | None:
    engine, X, y, n_variables = _resolve_refinement_arrays(payload)

    refiner = Refiner(simplipy_engine=engine, n_variables=n_variables)
    try:
        n_restarts = payload["n_restarts"]
        refiner.fit(
            expression=payload["tokens"],
            X=X,
            y=y.reshape(-1, 1),
            n_restarts=n_restarts,
            method="curve_fit_lm",
            p0_noise=None if n_restarts == 1 else "normal",
            p0_noise_kwargs={"loc": 0.0, "scale": payload["init_noise_scale"]},
            no_constants_error="ignore",
            converge_error="ignore",
        )
    except Exception:
        return None

    if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
        return None

    serialized_fits: list[tuple[np.ndarray, np.ndarray | None, float]] = []
    for constants, constants_cov, fit_loss in refiner._all_constants_values:
        cov_payload: np.ndarray | None
        if constants_cov is None or getattr(constants_cov, "size", 0) == 0:
            cov_payload = None
        else:
            cov_payload = np.asarray(constants_cov)
        serialized_fits.append((np.asarray(constants), cov_payload, float(fit_loss)))

    return {
        "cand_id": payload["cand_id"],
        "tokens": payload["tokens"],
        "log_prob": float(payload["log_prob"]),
        "fits": serialized_fits,
    }


class RASR(BaseEstimator, RegressorMixin):
    """sklearn-style interface for the trained RASRTransformer."""

    def __init__(
        self,
        model_path: str = "davidotte/RASR-94M-L",
        config_path: str | None = None,
        simplipy_engine_name: str = "dev_7-3",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        step: int | None = None,
        verbose: bool = False,
        num_candidates: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.95,
        max_len: int | None = None,
        refine_constants: bool = True,
        n_restarts: int = 4,
        init_noise_scale: float = 0.5,
        parsimony: float = 0.05,
        decode_batch_size: int = 128,
        refiner_workers: int | None = None,
        scoring: Literal["r2"] = "r2",
        standardize: bool = True,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.simplipy_engine_name = simplipy_engine_name
        self.step = step
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.verbose = verbose
        self.num_candidates = int(num_candidates)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.max_len = max_len
        self.refine_constants = bool(refine_constants)
        self.n_restarts = int(n_restarts)
        self.init_noise_scale = float(init_noise_scale)
        self.parsimony = float(parsimony)
        self.decode_batch_size = int(decode_batch_size)
        if refiner_workers is None:
            try:
                n_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                n_cpus = os.cpu_count() or 1
            self.refiner_workers = max(1, min(n_cpus - 2, self.num_candidates))
        else:
            self.refiner_workers = max(0, int(refiner_workers))
        if scoring != "r2":
            raise ValueError(f"scoring must be 'r2', got {scoring!r}")
        self.scoring = scoring
        self.standardize = bool(standardize)

        self.logger = logging.getLogger("RASR")
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Fork the refiner pool before CUDA is initialized; forking afterwards can deadlock.
        use_persistent_pool = (
            self.refiner_workers >= 1
            and self.device.startswith("cuda")
            and "fork" in mp.get_all_start_methods()
        )
        self._refiner_pool: ProcessPoolExecutor | None = None

        if use_persistent_pool:
            self.model, self.tokenizer, self._job_config = self._load_model(device="cpu")
        else:
            self.model, self.tokenizer, self._job_config = self._load_model()

        self.simplipy_engine = sp.SimpliPyEngine.load(self.simplipy_engine_name, install=True)
        self.operator_arity = dict(self.simplipy_engine.operator_arity)
        self.variables = set(self._job_config.tokenizer.variables)
        self._digit_tokens = self._build_digit_tokens()
        self._exponent_tokens = self._build_exponent_tokens()
        self._constant_token = (
            "<constant>"
            if getattr(self._job_config.tokenizer, "constant_strategy", "mantissa_exponent") == "placeholder"
            else None
        )
        self.n_variables = len(self._job_config.tokenizer.variables)

        if use_persistent_pool:
            self._init_persistent_pool()
            self.model = self.model.to(device=self.device, dtype=self.dtype)
            self.model.eval()

        self.results_: list[CandidateResult] = []
        self.best_result_: CandidateResult | None = None
        self.pareto_front_: list[CandidateResult] = []
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.variable_mapping: dict[str, str] = {}
        self.X_mean_: np.ndarray | None = None
        self.X_std_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_scale_: float = 1.0
        self._standardization_active_: bool = False

    def _build_digit_tokens(self) -> set[str]:
        tok_cfg = self._job_config.tokenizer
        digits = list(tok_cfg.digits)
        special_chars = list(tok_cfg.special_chars)
        return set(digits + [ch for ch in special_chars if len(ch) == 1])

    def _build_exponent_tokens(self) -> frozenset[str]:
        tok_cfg = self._job_config.tokenizer
        return frozenset(ch for ch in tok_cfg.special_chars if ch in ("e+", "e-"))

    def _load_model(self, device: str | None = None) -> tuple[RASRTransformer, EquationTokenizer, object]:
        model_dir = self.model_path if os.path.isdir(self.model_path) else snapshot_download(self.model_path)
        return load_rasr_model(
            model_path=model_dir,
            config_path=self.config_path or os.path.join(model_dir, "config.toml"),
            step=self.step,
            device=device if device is not None else self.device,
            dtype=self.dtype,
        )

    def _warm_refine_engine(self) -> None:
        """Warm up the engine before forking so workers inherit it; RNG state is preserved."""
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            n_vars = max(1, self.n_variables)
            X = np.ones((8, n_vars), dtype=np.float64)
            y = np.ones((8, 1), dtype=np.float64)
            warm_refiner = Refiner(simplipy_engine=self.simplipy_engine, n_variables=n_vars)
            warm_refiner.fit(
                expression=["*", "<constant>", "x1"],
                X=X,
                y=y,
                n_restarts=1,
                method="curve_fit_lm",
                p0_noise=None,
                p0_noise_kwargs={"loc": 0.0, "scale": 1.0},
                no_constants_error="ignore",
                converge_error="ignore",
            )
        except Exception:
            pass
        finally:
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)

    def _init_persistent_pool(self) -> None:
        """Fork the reusable refiner pool while BLAS is temporarily single-threaded."""
        blas_limiter: Any = None
        try:
            from threadpoolctl import threadpool_limits
            blas_limiter = threadpool_limits(limits=1)
        except Exception:
            pass
        try:
            self._warm_refine_engine()
            self._refiner_pool = ProcessPoolExecutor(
                max_workers=self.refiner_workers,
                context=mp.get_context("fork"),
                initializer=_init_persistent_refinement_worker,
                initargs=(self.simplipy_engine_name,),
            )
        finally:
            if blas_limiter is not None:
                blas_limiter.restore_original_limits()

    def close(self) -> None:
        """Shut down the persistent refiner pool."""
        pool = getattr(self, "_refiner_pool", None)
        if pool is not None:
            pool.shutdown()
            self._refiner_pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _to_numpy_2d(x: Any, *, name: str) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D. Got shape {arr.shape}.")
        return arr

    @staticmethod
    def _check_finite(arr: np.ndarray, *, name: str) -> None:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            raise ValueError(
                f"{name} contains {n_nan} NaN and {n_inf} infinite values. "
                "Impute or drop these rows before calling fit()."
            )

    def _pad_or_truncate_X(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] > self.n_variables:
            raise ValueError(
                f"Input has {X.shape[1]} variables, but model supports at most {self.n_variables}."
            )
        return X

    def _prepare_table_tensor(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Standardize X and y (skipped if already standardized or disabled) and build the model input."""
        if not self.standardize:
            self.X_mean_ = np.zeros(X.shape[1], dtype=np.float32)
            self.X_std_ = np.ones(X.shape[1], dtype=np.float32)
            self.y_mean_ = 0.0
            self.y_std_scale_ = 1.0
            self._standardization_active_ = False
            X_standardized = X
            y_standardized = y
        else:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0)
            x_std[x_std < 1e-8] = 1.0
            y_mean = float(y.mean())
            y_std_scale = float(y.std())
            if y_std_scale < 1e-8:
                y_std_scale = 1.0

            already_standardized = (
                np.all(np.abs(x_mean) < 0.1)
                and np.all(np.abs(x_std - 1.0) < 0.1)
                and abs(y_mean) < 0.1
                and abs(y_std_scale - 1.0) < 0.1
            )

            if already_standardized:
                self.X_mean_ = np.zeros(X.shape[1], dtype=np.float32)
                self.X_std_ = np.ones(X.shape[1], dtype=np.float32)
                self.y_mean_ = 0.0
                self.y_std_scale_ = 1.0
                self._standardization_active_ = False
                X_standardized = X
                y_standardized = y
            else:
                self.X_mean_ = x_mean
                self.X_std_ = x_std
                self.y_mean_ = y_mean
                self.y_std_scale_ = y_std_scale
                self._standardization_active_ = True
                X_standardized = ((X - self.X_mean_) / self.X_std_).astype(np.float32)
                y_standardized = ((y - self.y_mean_) / self.y_std_scale_).astype(np.float32)

        y_col = y_standardized.reshape(-1, 1)
        dataset = np.concatenate([y_col, X_standardized], axis=1).astype(np.float32)
        tbl = torch.from_numpy(dataset).unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
        return tbl, X_standardized, y_standardized

    def _decode_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        out: list[str] = []
        for idx in token_ids:
            token = self.tokenizer.index_to_token.get(int(idx))
            if token is not None:
                out.append(token)
        return out

    def _resolve_variable_mapping(
        self,
        X_raw: Any,
        variable_names: list[str] | dict[str, str] | Literal["auto"] | None,
        n_features_original: int,
    ) -> dict[str, str]:
        if variable_names is None:
            return {}
        if variable_names == "auto":
            if isinstance(X_raw, pd.DataFrame):
                return {f"x{i + 1}": str(col) for i, col in enumerate(X_raw.columns[:n_features_original])}
            return {}
        if isinstance(variable_names, list):
            return {f"x{i + 1}": str(name) for i, name in enumerate(variable_names[:n_features_original])}
        if isinstance(variable_names, dict):
            if isinstance(X_raw, pd.DataFrame):
                out: dict[str, str] = {}
                for i, col in enumerate(X_raw.columns[:n_features_original]):
                    out[f"x{i + 1}"] = str(variable_names.get(col, col))
                return out
            return {str(k): str(v) for k, v in variable_names.items()}
        raise TypeError("variable_names must be list, dict, 'auto', or None.")

    def _run_refinement(
        self,
        candidate_payloads: list[dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        perform_refinement: bool = True,
        converge_error: Literal["raise", "ignore", "print"] = "ignore",
    ) -> list[dict[str, Any]]:
        if not candidate_payloads:
            return []

        if not perform_refinement:
            out: list[dict[str, Any]] = []
            for payload in candidate_payloads:
                refiner = Refiner(self.simplipy_engine, n_variables=X.shape[1])
                try:
                    initial_values = refiner._initialize_expression(payload["tokens"], n_inputs=X.shape[1])
                except Exception:
                    continue

                if refiner.expression_lambda is None:
                    continue
                try:
                    with warnings.catch_warnings(), np.errstate(all="ignore"):
                        warnings.simplefilter("ignore")
                        if len(initial_values) == 0:
                            y_pred = refiner.expression_lambda(*X.T)
                        else:
                            y_pred = refiner.expression_lambda(*X.T, *initial_values)
                except Exception:
                    continue

                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
                if y_pred.size == 1 and X.shape[0] > 1:
                    y_pred = np.full((X.shape[0],), float(y_pred[0]))
                if y_pred.size != X.shape[0]:
                    y_pred = np.resize(y_pred, X.shape[0])

                constants_payload = np.asarray(initial_values, dtype=float)
                if not np.all(np.isfinite(y_pred)):
                    loss = np.nan
                else:
                    loss = float(np.mean((y_pred - y.reshape(-1)) ** 2))
                out.append(
                    {
                        "cand_id": payload["cand_id"],
                        "tokens": payload["tokens"],
                        "log_prob": payload["log_prob"],
                        "fits": [(constants_payload, np.array([]), loss)],
                    }
                )
            return out

        if self.refiner_workers == 0 or len(candidate_payloads) == 1:
            serial_out: list[dict[str, Any]] = []
            _init_refinement_worker(self.simplipy_engine_name, X, y, X.shape[1])
            for payload in tqdm(candidate_payloads, desc="Refining", smoothing=0.0, disable=not self.verbose):
                result = _refine_candidate_worker(payload)
                if result is not None:
                    serial_out.append(result)
                elif converge_error == "raise":
                    raise ConvergenceError("Constant refinement did not converge.")
            return serial_out

        results: list[dict[str, Any]] = []

        owns_executor = self._refiner_pool is None
        executor: ProcessPoolExecutor | None = None
        try:
            if not owns_executor:
                executor = self._refiner_pool
                task_payloads = [
                    {**payload, "X": X, "y": y, "n_variables": X.shape[1]} for payload in candidate_payloads
                ]
                future_to_payload = {
                    executor.submit(_refine_candidate_worker, payload): payload for payload in task_payloads
                }
            else:
                max_workers = min(self.refiner_workers, len(candidate_payloads))
                executor = ProcessPoolExecutor(
                    max_workers=max_workers,
                    context=mp.get_context("fork"),
                    initializer=_init_pooled_refinement_worker,
                    initargs=(self.simplipy_engine_name, X, y, X.shape[1]),
                )
                future_to_payload = {
                    executor.submit(_refine_candidate_worker, payload): payload for payload in candidate_payloads
                }

            for future in tqdm(
                as_completed(future_to_payload),
                total=len(future_to_payload),
                desc="Refining",
                smoothing=0.0,
                disable=not self.verbose,
            ):
                payload = future_to_payload[future]
                try:
                    result = future.result()
                except TerminatedWorkerError as e:
                    self.logger.warning(f"A refinement worker was killed by the OS: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Worker raised unexpected exception: {e}")
                    continue
                if result is not None:
                    results.append(result)
                elif converge_error == "raise":
                    raise ConvergenceError("Constant refinement did not converge.")
                elif converge_error == "print":
                    self.logger.warning(f"Refinement failed for candidate: {payload['tokens']}")
        finally:
            if owns_executor and executor is not None:
                executor.shutdown()
        return results

    def fit(
        self,
        X,
        y,
        variable_names: list[str] | dict[str, str] | Literal["auto"] | None = "auto",
        converge_error: Literal["raise", "ignore", "print"] = "ignore",
        simplify_expressions: bool = False,
        refine_candidates: bool | None = None,
        X_val=None,
        y_val=None,
    ):
        """Fit on (X, y); X_val/y_val (if given) are used for candidate selection."""
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both omitted.")

        X_raw = X
        X_np = self._to_numpy_2d(X, name="X").astype(np.float32)
        y_np = self._to_numpy_2d(y, name="y").astype(np.float32)
        if y_np.shape[1] != 1:
            raise ValueError("Only single-output targets are supported.")
        self._check_finite(X_np, name="X")
        self._check_finite(y_np, name="y")
        y_flat = y_np[:, 0]

        n_features_original = X_np.shape[1]
        self.variable_mapping = self._resolve_variable_mapping(X_raw, variable_names, n_features_original)
        X_model = self._pad_or_truncate_X(X_np)

        tbl, X_standardized, y_standardized = self._prepare_table_tensor(X_model, y_flat)

        X_val_standardized: np.ndarray | None = None
        y_val_standardized: np.ndarray | None = None
        if X_val is not None:
            X_val_np = self._to_numpy_2d(X_val, name="X_val").astype(np.float32)
            y_val_np = self._to_numpy_2d(y_val, name="y_val").astype(np.float32)[:, 0]
            self._check_finite(X_val_np, name="X_val")
            self._check_finite(y_val_np, name="y_val")
            if X_val_np.shape[1] != n_features_original:
                raise ValueError(
                    f"X_val has {X_val_np.shape[1]} features, but X has {n_features_original}."
                )
            X_val_model = self._pad_or_truncate_X(X_val_np)
            X_val_standardized = ((X_val_model - self.X_mean_) / self.X_std_).astype(np.float32)
            y_val_standardized = ((y_val_np - self.y_mean_) / self.y_std_scale_).astype(np.float32)

        sampled_ids, sampled_log_probs = self.model.sample(
            tbl=tbl,
            num_candidates=self.num_candidates,
            max_len=self.max_len,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            batch_size=self.decode_batch_size,
            verbose=self.verbose,
            extract_expressions=True,
            unique=True,
        )

        token_id_candidates = sampled_ids[0]
        log_prob_candidates = sampled_log_probs[0]

        candidate_payloads: list[dict[str, Any]] = []
        available_feature_count = X_model.shape[1]

        n_sampled = len(token_id_candidates)
        n_parse_failed = 0
        n_too_many_variables = 0
        parse_failure_examples: list[str] = []

        for token_ids, log_prob in zip(token_id_candidates, log_prob_candidates):
            decoded = self._decode_ids_to_tokens(token_ids)
            parsed_tokens = parse_prefix_tokens(
                decoded,
                operator_arity=self.operator_arity,
                variables=self.variables,
                digit_tokens=self._digit_tokens,
                exponent_tokens=self._exponent_tokens,
                constant_token=self._constant_token,
            )
            if parsed_tokens is None:
                n_parse_failed += 1
                if len(parse_failure_examples) < 5:
                    parse_failure_examples.append(" ".join(decoded))
                continue
            req = required_variable_count(parsed_tokens)
            if req > available_feature_count:
                n_too_many_variables += 1
                continue
            if simplify_expressions:
                try:
                    simplified_tokens = self.simplipy_engine.simplify(
                        parsed_tokens, inplace=False, max_pattern_length=4
                    )
                except Exception:
                    simplified_tokens = parsed_tokens
            else:
                simplified_tokens = parsed_tokens

            candidate_payloads.append(
                {
                    "cand_id": len(candidate_payloads),
                    "tokens": simplified_tokens,
                    "log_prob": float(log_prob),
                    "n_restarts": self.n_restarts,
                    "init_noise_scale": self.init_noise_scale,
                }
            )

        self.logger.info(
            "fit(): sampled %d candidates -> %d parsed (%d parse failures, %d needed too many variables)",
            n_sampled, len(candidate_payloads), n_parse_failed, n_too_many_variables,
        )
        if parse_failure_examples:
            self.logger.info("  example raw decoded token streams that failed to parse:")
            for example in parse_failure_examples:
                self.logger.info("    %s", example)
        if candidate_payloads:
            self.logger.info("  example parsed expressions (pre-refinement):")
            for payload in candidate_payloads[:5]:
                self.logger.info(
                    "    %s",
                    self.get_expression_from_tokens(payload["tokens"], return_prefix=False),
                )

        perform_refinement = self.refine_constants if refine_candidates is None else bool(refine_candidates)

        refined_payloads = self._run_refinement(
            candidate_payloads=candidate_payloads,
            X=X_standardized,
            y=y_standardized,
            perform_refinement=perform_refinement,
            converge_error=converge_error,
        )

        self.logger.info(
            "fit(): refinement kept %d/%d candidates (perform_refinement=%s)",
            len(refined_payloads), len(candidate_payloads), perform_refinement,
        )
        if candidate_payloads and not refined_payloads:
            self.logger.info("  example expressions that entered refinement (all were dropped):")
            for payload in candidate_payloads[:5]:
                self.logger.info(
                    "    %s",
                    self.get_expression_from_tokens(payload["tokens"], return_prefix=False),
                )

        results: list[CandidateResult] = []
        n_invalid_fit = 0
        n_nonfinite_pred = 0
        n_nonfinite_score = 0
        n_lambda_failed = 0

        for payload in refined_payloads:
            refiner = Refiner.from_serialized(
                simplipy_engine=self.simplipy_engine,
                n_variables=X_standardized.shape[1],
                expression=payload["tokens"],
                n_inputs=X_standardized.shape[1],
                fits=payload["fits"],
            )
            if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
                n_invalid_fit += 1
                continue
            y_pred_standardized = refiner.predict(X_standardized, nth_best_constants=0).reshape(-1)
            if not np.all(np.isfinite(y_pred_standardized)):
                n_nonfinite_pred += 1
                continue

            candidate_r2 = float(r2_score(y_standardized, y_pred_standardized))
            complexity = len(payload["tokens"])

            if X_val_standardized is not None:
                y_val_pred_standardized = refiner.predict(
                    X_val_standardized, nth_best_constants=0
                ).reshape(-1)
                if np.all(np.isfinite(y_val_pred_standardized)):
                    val_r2 = float(r2_score(y_val_standardized, y_val_pred_standardized))
                else:
                    val_r2 = float("nan")
            else:
                val_r2 = float("nan")

            best_fit = refiner._all_constants_values[0] if refiner._all_constants_values else None
            constants_arr = best_fit[0] if best_fit is not None else np.array([])

            effective_r2 = val_r2 if X_val_standardized is not None else candidate_r2
            score = effective_r2 - self.parsimony * complexity

            if not np.isfinite(score):
                n_nonfinite_score += 1
                continue

            constants_iter = iter(constants_arr.tolist())
            materialized_tokens = [
                str(next(constants_iter)) if t == "<constant>" else t for t in payload["tokens"]
            ]

            try:
                fn, _ = expression_to_lambda(
                    materialized_tokens,
                    simplipy_engine=self.simplipy_engine,
                    all_variables=list(self._job_config.tokenizer.variables),
                )
            except Exception as e:
                n_lambda_failed += 1
                self.logger.info(
                    "  lambda build failed for %s: %s",
                    self.get_expression_from_tokens(payload["tokens"], return_prefix=False), e,
                )
                continue

            results.append(
                CandidateResult(
                    refiner=refiner,
                    tokens=payload["tokens"],
                    expression_string=self.get_expression_from_tokens(payload["tokens"], return_prefix=False),
                    score=score,
                    r2=candidate_r2,
                    log_prob=float(payload["log_prob"]),
                    y_pred=y_pred_standardized,
                    lambda_function=fn,
                    constants=constants_arr.tolist(),
                    complexity=complexity,
                    val_r2=val_r2,
                )
            )

        self.logger.info(
            "fit(): post-refinement filtering -- invalid_fit=%d, nonfinite_pred=%d, "
            "nonfinite_score=%d, lambda_failed=%d, valid=%d",
            n_invalid_fit, n_nonfinite_pred, n_nonfinite_score, n_lambda_failed, len(results),
        )

        if not results:
            raise RuntimeError(
                "No valid candidate expression found. "
                f"sampled={n_sampled}, parsed={len(candidate_payloads)} "
                f"(parse_failed={n_parse_failed}, too_many_variables={n_too_many_variables}), "
                f"refined={len(refined_payloads)}, "
                f"post_refinement_drops(invalid_fit={n_invalid_fit}, "
                f"nonfinite_pred={n_nonfinite_pred}, nonfinite_score={n_nonfinite_score}, "
                f"lambda_failed={n_lambda_failed}). "
                "Set verbose=True on RASR(...) to see per-stage example expressions."
            )

        results.sort(key=lambda r: r.score, reverse=True)

        self.results_ = results
        self.best_result_ = results[0]
        self.pareto_front_ = _pareto_front(results, use_val=X_val_standardized is not None)
        self.X_train_ = X_model
        self.y_train_ = y_flat

        for i, result in enumerate(self.results_):
            result.expression_string = self.get_expression(nth_best=i)

        return self

    def predict(self, X, nth_best_beam: int = 0, nth_best_constants: int = 0) -> np.ndarray:
        if self.best_result_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        if nth_best_beam < 0 or nth_best_beam >= len(self.results_):
            raise IndexError(f"nth_best_beam={nth_best_beam} is out of range.")
        X_np = self._to_numpy_2d(X, name="X").astype(np.float32)
        X_model = self._pad_or_truncate_X(X_np)
        X_standardized = ((X_model - self.X_mean_) / self.X_std_).astype(np.float32)

        y_pred_standardized = self.results_[nth_best_beam].refiner.predict(
            X_standardized, nth_best_constants=nth_best_constants
        ).reshape(-1)
        return y_pred_standardized * self.y_std_scale_ + self.y_mean_

    def get_best_candidates(self, k: int = 1) -> list[dict[str, Any]]:
        if self.best_result_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        if k <= 0:
            raise ValueError("k must be > 0")
        top = self.results_[:k]
        return [
            {
                "expression_tokens": result.tokens,
                "expression": self.get_expression_from_tokens(result.tokens),
                "score": result.score,
                "r2_score": result.r2,
                "val_r2": result.val_r2,
                "log_prob": result.log_prob,
                "constants": result.constants,
            }
            for result in top
        ]

    def get_expression_from_tokens(
        self,
        tokens: list[str],
        *,
        return_prefix: bool = False,
        precision: int = 4,
        map_variables: bool = True,
    ) -> str:
        out_tokens: list[str] = []
        for token in tokens:
            if is_numeric_string(token):
                out_tokens.append(f"{float(token):.{precision}g}")
            elif map_variables and token in self.variable_mapping:
                out_tokens.append(self.variable_mapping[token])
            else:
                out_tokens.append(token)

        if return_prefix:
            return " ".join(out_tokens)

        try:
            expr = self.simplipy_engine.prefix_to_infix(list(tokens), realization=False)
            if map_variables and self.variable_mapping:
                for old_name, new_name in self.variable_mapping.items():
                    expr = re.sub(rf"\b{re.escape(old_name)}\b", new_name, expr)
            return expr
        except Exception:
            return " ".join(out_tokens)

    def get_expression(
        self,
        nth_best: int = 0,
        *,
        nth_best_constants: int = 0,
        return_prefix: bool = False,
        precision: int = 2,
        map_variables: bool = True,
        original_space: bool = True,
    ) -> str:
        """Return the expression, in the original data scale unless original_space=False."""
        if self.best_result_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        if nth_best < 0 or nth_best >= len(self.results_):
            raise IndexError(f"nth_best={nth_best} is out of range.")

        result = self.results_[nth_best]

        if original_space and self._standardization_active_ and self.X_mean_ is not None:
            best_fit = result.refiner._all_constants_values[nth_best_constants]
            fitted_constants = np.asarray(best_fit[0], dtype=float)
            new_tokens, _ = remap_to_original_space(
                result.tokens,
                fitted_constants,
                self.X_mean_,
                self.X_std_,
                self.y_mean_,
                self.y_std_scale_,
                dict(self.operator_arity),
                self.variables,
                precision=precision,
            )
            vm = self.variable_mapping if map_variables else None
            if return_prefix:
                out_tokens = [vm.get(t, t) if vm else t for t in new_tokens]
                return " ".join(out_tokens)
            try:
                expr = self.simplipy_engine.prefix_to_infix(list(new_tokens), realization=False)
                if vm:
                    for old, new in vm.items():
                        expr = re.sub(rf"\b{re.escape(old)}\b", new, expr)
                return expr
            except Exception:
                out_tokens = [vm.get(t, t) if vm else t for t in new_tokens]
                return " ".join(out_tokens)

        transformed = result.refiner.transform(
            result.tokens,
            nth_best_constants=nth_best_constants,
            return_prefix=return_prefix,
            precision=precision,
            variable_mapping=self.variable_mapping if map_variables else None,
            realization=False,
        )
        if isinstance(transformed, list):
            return " ".join(transformed)
        return transformed
