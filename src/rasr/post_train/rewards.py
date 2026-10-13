from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import numpy as np
import simplipy as sp
import torch
from sklearn.preprocessing import StandardScaler

from src.rasr.inference.expression_parser import (
    expression_to_lambda,
    parse_prefix_tokens,
    required_variable_count,
)
from src.rasr.inference.metrics import nrmse_reward, r2_score
from src.rasr.inference.refiner import Refiner
from src.rasr.post_train.constants import N_SELECTION_ROWS, N_VALIDATION_ROWS

_GLOBAL_ENGINE: sp.SimpliPyEngine | None = None


class InvalidExpressionError(ValueError):
    pass


class ConstantFittingFailedError(RuntimeError):
    pass


def _log_ratio_metric(log_metric: Any, name: str, count: int, total: int) -> None:
    if not callable(log_metric):
        return
    if total <= 0:
        return
    log_metric(name, float(count) / float(total))


def _init_worker(engine_name: str) -> None:
    global _GLOBAL_ENGINE
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(all="ignore")
    _GLOBAL_ENGINE = sp.SimpliPyEngine.load(engine_name, install=True)


def _safe_predict(fn, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        y_pred = fn(*X.T)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_pred.size == 1 and X.shape[0] > 1:
        y_pred = np.full((X.shape[0],), float(y_pred[0]))
    if y_pred.size != X.shape[0]:
        y_pred = np.resize(y_pred, X.shape[0])
    return y_pred


def _safe_predict_constant(fn, n_rows: int) -> np.ndarray:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        y_pred = fn()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_pred.size == 0:
        raise ValueError("Constant expression produced an empty prediction.")
    value = float(y_pred[0])
    return np.full((n_rows,), value, dtype=float)


def _parse_and_prepare_sample(
    token_stream: list[str],
    sample_tbl: np.ndarray,
    *,
    operator_arity: dict[str, int],
    variables: set[str],
    digit_tokens: set[str],
    exponent_tokens: frozenset[str] = frozenset(),
    simplipy_engine: sp.SimpliPyEngine,
    all_variables: list[str],
    standardize_dataset: bool,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parsed = parse_prefix_tokens(
        token_stream,
        operator_arity=operator_arity,
        variables=variables,
        digit_tokens=digit_tokens,
        exponent_tokens=exponent_tokens,
    )
    if parsed is None:
        raise InvalidExpressionError("Could not parse a valid expression from completion tokens.")

    n_rows = int(sample_tbl.shape[0])
    if n_rows <= N_VALIDATION_ROWS:
        raise ValueError(
            f"Expected more than {N_VALIDATION_ROWS} rows per sample, but got {n_rows}."
        )

    split_idx = n_rows - N_VALIDATION_ROWS
    y_all = sample_tbl[:, 0]
    X_all = sample_tbl[:, 1:]

    y_train = np.asarray(y_all[:split_idx], dtype=float)
    X_train = np.asarray(X_all[:split_idx], dtype=float)
    y_val = np.asarray(y_all[split_idx:], dtype=float)
    X_val = np.asarray(X_all[split_idx:], dtype=float)

    # Selection rows: last N_SELECTION_ROWS of the training portion, used for select@k evaluation.
    sel_start = max(0, split_idx - N_SELECTION_ROWS)
    y_sel = np.asarray(y_all[sel_start:split_idx], dtype=float)
    X_sel = np.asarray(X_all[sel_start:split_idx], dtype=float)

    finite_train = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
    finite_val = np.isfinite(y_val) & np.all(np.isfinite(X_val), axis=1)
    y_train = y_train[finite_train]
    X_train = X_train[finite_train]
    y_val = y_val[finite_val]
    X_val = X_val[finite_val]
    if y_train.size == 0 or X_train.shape[0] == 0:
        raise ValueError("No finite training rows available after filtering.")
    if y_val.size == 0 or X_val.shape[0] == 0:
        raise ValueError("No finite validation rows available after filtering.")

    if y_sel.size > 0:
        finite_sel = np.isfinite(y_sel) & np.all(np.isfinite(X_sel), axis=1)
        y_sel = y_sel[finite_sel]
        X_sel = X_sel[finite_sel]

    needed_vars = required_variable_count(parsed)
    if needed_vars > X_train.shape[1] or needed_vars > X_val.shape[1]:
        raise InvalidExpressionError(
            f"Expression requires {needed_vars} variables, but sample has {X_train.shape[1]} features."
        )

    X_train_eval = X_train[:, :needed_vars]
    X_val_eval = X_val[:, :needed_vars]
    X_sel_eval = X_sel[:, :needed_vars] if y_sel.size > 0 else np.empty((0, needed_vars), dtype=float)

    if standardize_dataset:
        if X_train_eval.shape[1] > 0:
            x_scaler = StandardScaler()
            X_train_eval = x_scaler.fit_transform(X_train_eval)
            X_val_eval = x_scaler.transform(X_val_eval)
            if X_sel_eval.shape[0] > 0:
                X_sel_eval = x_scaler.transform(X_sel_eval)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
        if y_sel.size > 0:
            y_sel = y_scaler.transform(y_sel.reshape(-1, 1)).reshape(-1)

    return parsed, X_train_eval, y_train, X_val_eval, y_val, X_sel_eval, y_sel


def _refine_worker(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    if _GLOBAL_ENGINE is None:
        raise RuntimeError("Reward worker engine was not initialized.")

    expression = payload["expression"]
    X_train = np.asarray(payload["X_train"], dtype=float)
    y_train = np.asarray(payload["y_train"], dtype=float)
    X_val = np.asarray(payload["X_val"], dtype=float)
    y_val = np.asarray(payload["y_val"], dtype=float)
    X_sel = np.asarray(payload["X_sel"], dtype=float)
    y_sel = np.asarray(payload["y_sel"], dtype=float)

    refiner = Refiner(_GLOBAL_ENGINE, n_variables=X_train.shape[1])
    try:
        refiner.fit(
            expression=expression,
            X=X_train,
            y=y_train.reshape(-1, 1),
            n_restarts=payload["n_restarts"],
            method="curve_fit_lm",
            p0_noise="normal",
            p0_noise_kwargs={"loc": 0.0, "scale": payload["init_noise_scale"]},
            no_constants_error="ignore",
            converge_error="ignore",
        )
    except Exception as exc:
        raise ConstantFittingFailedError("Constant fitting raised an exception.") from exc
    if not refiner.valid_fit:
        raise ConstantFittingFailedError("Constant fitting did not produce a valid fit.")

    try:
        y_val_pred = refiner.predict(X_val, nth_best_constants=0).reshape(-1)
        y_train_pred = refiner.predict(X_train, nth_best_constants=0).reshape(-1)
    except Exception as exc:
        raise ConstantFittingFailedError("Prediction with fitted constants failed.") from exc
    if not np.all(np.isfinite(y_val_pred)):
        raise ConstantFittingFailedError("Prediction with fitted constants produced non-finite outputs.")

    val_r2 = max(float(r2_score(y_val, y_val_pred)), -1.0)
    if not np.isfinite(val_r2):
        val_r2 = -1.0
    val_nrmse_rw = nrmse_reward(y_val, y_val_pred)
    train_r2 = max(float(r2_score(y_train, y_train_pred)), -1.0) if np.all(np.isfinite(y_train_pred)) else float("nan")

    sel_r2 = float("nan")
    if X_sel.shape[0] > 0:
        try:
            y_sel_pred = refiner.predict(X_sel, nth_best_constants=0).reshape(-1)
            if np.all(np.isfinite(y_sel_pred)):
                sel_r2 = max(float(r2_score(y_sel, y_sel_pred)), -1.0)
                if not np.isfinite(sel_r2):
                    sel_r2 = float("nan")
        except Exception:
            pass

    return val_nrmse_rw, val_r2, train_r2, sel_r2


class NRMSERewardFn:
    def __init__(
        self,
        tokenizer,
        job_config,
        simplipy_engine_name: str = "dev_7-3",
        refine_constants: bool = False,
        standardize_dataset: bool = False,
        n_restarts: int = 4,
        init_noise_scale: float = 1.0,
        refinement_workers: int | None = None,
        refinement_timeout: float = 100.0,
        curriculum_init_noise_scale: bool = False,
        max_steps: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.job_config = job_config
        self.refine_constants = bool(refine_constants)
        self.standardize_dataset = bool(standardize_dataset)
        self.n_restarts = int(n_restarts)
        self.init_noise_scale = float(init_noise_scale)
        self.refinement_timeout = float(refinement_timeout)
        self.curriculum_init_noise_scale = bool(curriculum_init_noise_scale)
        self.max_steps = max(1, int(max_steps))
        self.simplipy_engine_name = simplipy_engine_name

        self.simplipy_engine = sp.SimpliPyEngine.load(self.simplipy_engine_name, install=True)
        self.operator_arity = dict(self.simplipy_engine.operator_arity)
        self.all_variables = list(job_config.tokenizer.variables)
        self.variables = set(self.all_variables)
        self.digit_tokens = set(job_config.tokenizer.digits) | {
            char for char in job_config.tokenizer.special_chars if len(char) == 1
        }
        self.exponent_tokens = frozenset(
            char for char in job_config.tokenizer.special_chars if char in ("e+", "e-")
        )

        workers = refinement_workers
        if workers is None:
            workers = max(1, min(8, (mp.cpu_count() or 1)))

        self._executor: ProcessPoolExecutor | None = None
        if self.refine_constants and workers > 0:
            kwargs: dict[str, Any] = {
                "max_workers": int(workers),
                "initializer": _init_worker,
                "initargs": (self.simplipy_engine_name,),
            }
            if "fork" in mp.get_all_start_methods():
                kwargs["mp_context"] = mp.get_context("fork")
            self._executor = ProcessPoolExecutor(**kwargs)

    @property
    def __name__(self) -> str:
        return "nrmse_reward"

    def _decode_to_tokens(self, ids: list[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            ids,
            skip_special_tokens=True,
            stop_at_eoe=True,
        )

    def _score_without_refinement(self, expression: list[str], X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Returns (r2, nrmse_reward_val)."""
        used_variables = [v for v in self.all_variables if v in expression]
        try:
            if len(used_variables) == 0:
                executable = self.simplipy_engine.operators_to_realizations(expression)
                code_string = self.simplipy_engine.prefix_to_infix(executable, realization=True)
                code = sp.codify(code_string, [])
                fn = self.simplipy_engine.code_to_lambda(code)
                y_pred = _safe_predict_constant(fn, n_rows=y.shape[0])
            else:
                fn, _ = expression_to_lambda(
                    expression_tokens=expression,
                    simplipy_engine=self.simplipy_engine,
                    all_variables=self.all_variables,
                )
                y_pred = _safe_predict(fn, X)
        except Exception as exc:
            raise InvalidExpressionError("Expression evaluation failed.") from exc
        if not np.all(np.isfinite(y_pred)):
            return -1.0, 0.0
        score_r2 = max(float(r2_score(y, y_pred)), -1.0)
        if not np.isfinite(score_r2):
            score_r2 = -1.0
        score_nrmse = nrmse_reward(y, y_pred)
        return score_r2, score_nrmse

    def __call__(
        self,
        completions,
        completion_ids,
        tbl: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, list[float]]:
        if isinstance(tbl, torch.Tensor):
            tbl_np = tbl.detach().cpu().numpy()
        elif isinstance(tbl, list) and tbl and isinstance(tbl[0], torch.Tensor):
            tbl_np = [t.detach().cpu().numpy() for t in tbl]
        else:
            tbl_np = np.asarray(tbl)

        log_metric = kwargs.get("log_metric")

        _pid = os.getpid()
        if self.curriculum_init_noise_scale:
            trainer_state = kwargs.get("trainer_state")
            step = getattr(trainer_state, "global_step", 0) if trainer_state is not None else 0
            noise_scale = self.init_noise_scale * max(0.0, 1.0 - step / self.max_steps)
        else:
            noise_scale = self.init_noise_scale

        scores: list[float] = [0.0] * len(tbl_np)
        val_r2_scores: list[float] = [-1.0] * len(tbl_np)
        train_scores: list[float] = [float("nan")] * len(tbl_np)
        sel_r2_scores: list[float] = [float("nan")] * len(tbl_np)
        refinement_payloads: list[tuple[int, dict[str, Any]]] = []
        invalid_expression_count = 0
        constant_fitting_failed_count = 0

        _t_parse_start = time.perf_counter()
        for idx, sample_tbl in enumerate(tbl_np):
            ids_row = completion_ids[idx]
            if isinstance(ids_row, torch.Tensor):
                ids_row = ids_row.detach().cpu().tolist()
            else:
                ids_row = [int(value) for value in ids_row]

            token_stream = self._decode_to_tokens(ids_row)
            _t_sample = time.perf_counter()
            try:
                parsed, X_train, y_train, X_val, y_val, X_sel, y_sel = _parse_and_prepare_sample(
                    token_stream,
                    sample_tbl,
                    operator_arity=self.operator_arity,
                    variables=self.variables,
                    digit_tokens=self.digit_tokens,
                    exponent_tokens=self.exponent_tokens,
                    simplipy_engine=self.simplipy_engine,
                    all_variables=self.all_variables,
                    standardize_dataset=self.standardize_dataset,
                )
            except InvalidExpressionError:
                invalid_expression_count += 1
                continue
            _dt = time.perf_counter() - _t_sample
            if _dt > 1.0:
                print(
                    f"[REWARD][pid={_pid}] slow parse idx={idx} tokens={len(token_stream)} "
                    f"expr_len={len(parsed)} dt={_dt:.2f}s",
                    file=sys.stderr, flush=True,
                )

            if self.refine_constants:
                refinement_payloads.append(
                    (
                        idx,
                        {
                            "expression": parsed,
                            "X_train": X_train,
                            "y_train": y_train,
                            "X_val": X_val,
                            "y_val": y_val,
                            "X_sel": X_sel,
                            "y_sel": y_sel,
                            "n_restarts": self.n_restarts,
                            "init_noise_scale": noise_scale,
                        },
                    )
                )
            else:
                try:
                    train_r2, _ = self._score_without_refinement(parsed, X_train, y_train)
                    train_scores[idx] = train_r2
                except InvalidExpressionError:
                    pass
                try:
                    val_r2, val_nrmse_rw = self._score_without_refinement(parsed, X_val, y_val)
                except InvalidExpressionError:
                    invalid_expression_count += 1
                    continue
                val_r2_scores[idx] = val_r2
                scores[idx] = val_nrmse_rw
                if X_sel.shape[0] > 0:
                    try:
                        sel_r2, _ = self._score_without_refinement(parsed, X_sel, y_sel)
                        sel_r2_scores[idx] = sel_r2
                    except (InvalidExpressionError, Exception):
                        pass

        _t_parse_total = time.perf_counter() - _t_parse_start
        print(
            f"[REWARD][pid={_pid}] parse_loop done  n={len(tbl_np)} valid={len(refinement_payloads)} "
            f"dt={_t_parse_total:.2f}s",
            file=sys.stderr, flush=True,
        )

        if self.refine_constants:
            _t_futures_start = time.perf_counter()
            future_to_idx = {
                self._executor.submit(_refine_worker, payload): idx
                for idx, payload in refinement_payloads
            }
            deadline = time.monotonic() + self.refinement_timeout
            timed_out_count = 0
            for future, idx in future_to_idx.items():
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    val_nrmse_rw, val_r2, train_r2, sel_r2 = future.result(timeout=remaining)
                    scores[idx] = val_nrmse_rw
                    val_r2_scores[idx] = val_r2
                    train_scores[idx] = train_r2
                    sel_r2_scores[idx] = sel_r2
                except (ConstantFittingFailedError, FutureTimeoutError) as _e:
                    if isinstance(_e, FutureTimeoutError):
                        timed_out_count += 1
                    constant_fitting_failed_count += 1
            print(
                f"[REWARD][pid={_pid}] futures done  n={len(future_to_idx)} timed_out={timed_out_count} "
                f"dt={time.perf_counter() - _t_futures_start:.2f}s",
                file=sys.stderr, flush=True,
            )

        _log_ratio_metric(
            log_metric,
            "rewards/nrmse_reward/invalid_expression_ratio",
            invalid_expression_count,
            len(tbl_np),
        )
        if self.refine_constants:
            _log_ratio_metric(
                log_metric,
                "rewards/nrmse_reward/constant_fitting_failed_ratio",
                constant_fitting_failed_count,
                len(tbl_np),
            )

        for idx, score in enumerate(scores):
            if not np.isfinite(score):
                scores[idx] = 0.0

        return {"reward": scores, "val_r2": val_r2_scores, "train_r2": train_scores, "sel_r2": sel_r2_scores}

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


class ComplexityRewardFn:
    def __init__(
        self,
        tokenizer,
        job_config,
        simplipy_engine_name: str = "dev_7-3",
        standardize_dataset: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.all_variables = list(job_config.tokenizer.variables)
        self.standardize_dataset = bool(standardize_dataset)

        simplipy_engine = sp.SimpliPyEngine.load(simplipy_engine_name, install=True)
        self.operator_arity = dict(simplipy_engine.operator_arity)
        self.variables = set(self.all_variables)
        self.digit_tokens = set(job_config.tokenizer.digits) | {
            char for char in job_config.tokenizer.special_chars if len(char) == 1
        }
        self.exponent_tokens = frozenset(
            char for char in job_config.tokenizer.special_chars if char in ("e+", "e-")
        )
        self.simplipy_engine = simplipy_engine

    @property
    def __name__(self) -> str:
        return "complexity_reward"

    def _decode_to_tokens(self, ids: list[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            ids,
            skip_special_tokens=True,
            stop_at_eoe=True,
        )

    def __call__(
        self,
        completions,
        completion_ids,
        tbl: torch.Tensor,
        **kwargs: object,
    ) -> list[float]:
        del completions

        if isinstance(tbl, torch.Tensor):
            tbl_np = tbl.detach().cpu().numpy()
        elif isinstance(tbl, list) and tbl and isinstance(tbl[0], torch.Tensor):
            tbl_np = [t.detach().cpu().numpy() for t in tbl]
        else:
            tbl_np = np.asarray(tbl)

        log_metric = kwargs.get("log_metric")

        scores: list[float] = [-1.0] * len(tbl_np)
        invalid_expression_count = 0

        for idx, sample_tbl in enumerate(tbl_np):
            ids_row = completion_ids[idx]
            if isinstance(ids_row, torch.Tensor):
                ids_row = ids_row.detach().cpu().tolist()
            else:
                ids_row = [int(value) for value in ids_row]

            token_stream = self._decode_to_tokens(ids_row)
            try:
                expression, _, _, _, _, _, _ = _parse_and_prepare_sample(
                    token_stream,
                    sample_tbl,
                    operator_arity=self.operator_arity,
                    variables=self.variables,
                    digit_tokens=self.digit_tokens,
                    exponent_tokens=self.exponent_tokens,
                    simplipy_engine=self.simplipy_engine,
                    all_variables=self.all_variables,
                    standardize_dataset=self.standardize_dataset,
                )
            except InvalidExpressionError:
                invalid_expression_count += 1
                continue

            scores[idx] = -float(len(expression))

        _log_ratio_metric(
            log_metric,
            "rewards/complexity_reward/invalid_expression_ratio",
            invalid_expression_count,
            len(tbl_np),
        )

        return scores
