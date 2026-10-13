"""Modified version of the constant refiner in https://github.com/psaegert/flash-ansr/tree/main."""
from __future__ import annotations

import importlib
import logging
import time
import warnings
from typing import Any, Callable, Literal

import numpy as np
import simplipy as sp
import torch
from scipy.optimize import OptimizeWarning, curve_fit
from simplipy import SimpliPyEngine

logger = logging.getLogger(__name__)


class ConvergenceError(Exception):
    pass


def _is_numeric_token(token: str) -> bool:
    if token == "<constant>":
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def _require_input_set_width(X: np.ndarray | torch.Tensor, n_variables: int):
    if X.shape[1] != n_variables:
        raise ValueError(
            f"Expected input width {n_variables}, but received {X.shape[1]}. "
            "Strict mode does not pad or truncate input columns."
        )
    return X


class Refiner:
    """Refine constants for a prefix expression. Only method='curve_fit_lm' is supported."""

    def __init__(self, simplipy_engine: SimpliPyEngine, n_variables: int):
        self.simplipy_engine = simplipy_engine
        self.n_variables = n_variables
        self.import_modules()

        self.input_expression: list[str] = []
        self.executable_prefix_expression: list[str] = []
        self.prefix_expression_with_constants: list[str] = []
        self.constants_symbols: list[str] = []
        self.code_string: str = ""
        self.expression_code: Callable | None = None
        self.expression_lambda: Callable | None = None

        self.loss = np.inf
        self.valid_fit = False
        self._all_constants_values: list[tuple[np.ndarray, np.ndarray, float]] = []

    def import_modules(self) -> None:
        for module in getattr(self.simplipy_engine, "modules", []):
            if module not in globals():
                globals()[module] = importlib.import_module(module)

    def _identify_constants(self, expression: list[str]) -> tuple[list[str], list[str], list[float]]:
        with_constants: list[str] = []
        symbols: list[str] = []
        initial_values: list[float] = []
        counter = 0
        operator_arity = self.simplipy_engine.operator_arity
        variables = {f"x{i + 1}" for i in range(self.n_variables)}

        for token in expression:
            if token in operator_arity or token in variables:
                with_constants.append(token)
                continue
            if _is_numeric_token(token):
                symbol = f"C_{counter}"
                counter += 1
                with_constants.append(symbol)
                symbols.append(symbol)
                if token == "<constant>":
                    initial_values.append(0.0)
                else:
                    initial_values.append(float(token))
                continue
            with_constants.append(token)

        return with_constants, symbols, initial_values

    def _initialize_expression(self, expression: list[str], n_inputs: int) -> list[float]:
        if int(n_inputs) != int(self.n_variables):
            raise ValueError(
                f"Expected {self.n_variables} input variables, but got n_inputs={n_inputs}. "
                "Strict mode requires exact input arity."
            )

        self.input_expression = list(expression)

        logger.debug("refiner: prefix_to_infix start %s", expression)
        t0 = time.monotonic()
        self.executable_prefix_expression = self.simplipy_engine.operators_to_realizations(self.input_expression)
        self.prefix_expression_with_constants, self.constants_symbols, initial_values = self._identify_constants(
            self.input_expression
        )
        self.code_string = self.simplipy_engine.prefix_to_infix(
            self.prefix_expression_with_constants, realization=True
        )
        logger.debug("refiner: prefix_to_infix done %.2fs", time.monotonic() - t0)

        variable_names = [f"x{i + 1}" for i in range(int(self.n_variables))]

        logger.debug("refiner: code_to_lambda start")
        t0 = time.monotonic()
        expression_code = sp.codify(self.code_string, variable_names + self.constants_symbols)
        self.expression_code = expression_code
        self.expression_lambda = self.simplipy_engine.code_to_lambda(expression_code)
        logger.debug("refiner: code_to_lambda done %.2fs", time.monotonic() - t0)

        return initial_values

    def _assign_fits(self, fits: list[tuple[np.ndarray, np.ndarray | None, float]]) -> None:
        expected = len(self.constants_symbols)
        parsed: list[tuple[np.ndarray, np.ndarray, float]] = []
        for constants, cov, loss in fits:
            c = np.asarray(constants, dtype=float)
            if c.shape[0] != expected:
                continue
            cov_arr = np.asarray(cov) if cov is not None else np.asarray([], dtype=float)
            parsed.append((c, cov_arr, float(loss)))
        self._all_constants_values = sorted(parsed, key=lambda item: item[-1])
        self.valid_fit = any(np.isfinite(item[-1]) for item in self._all_constants_values)
        self.loss = float(self._all_constants_values[0][-1]) if self._all_constants_values else np.inf

    @classmethod
    def from_serialized(
        cls,
        simplipy_engine: SimpliPyEngine,
        n_variables: int,
        expression: list[str],
        n_inputs: int,
        fits: list[tuple[np.ndarray, np.ndarray | None, float]],
    ) -> "Refiner":
        refiner = cls(simplipy_engine=simplipy_engine, n_variables=n_variables)
        refiner._initialize_expression(expression, n_inputs=n_inputs)
        refiner._assign_fits(fits)
        return refiner

    def fit(
        self,
        expression: list[str],
        X: np.ndarray,
        y: np.ndarray,
        p0: np.ndarray | None = None,
        p0_noise: Literal["uniform", "normal"] | None = "normal",
        p0_noise_kwargs: dict | None = None,
        n_restarts: int = 8,
        method: Literal["curve_fit_lm"] = "curve_fit_lm",
        no_constants_error: Literal["raise", "ignore"] = "ignore",
        optimizer_kwargs: dict | None = None,
        converge_error: Literal["raise", "ignore"] = "ignore",
    ) -> "Refiner":
        if method != "curve_fit_lm":
            raise ValueError("Only method='curve_fit_lm' is supported.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        X = np.asarray(_require_input_set_width(X, self.n_variables), dtype=float)

        initial_values = self._initialize_expression(expression, n_inputs=X.shape[1])

        def pred_function(X_in: np.ndarray, *constants: np.ndarray | None) -> np.ndarray:
            if self.expression_lambda is None:
                raise RuntimeError("Expression lambda is not initialized.")
            if len(constants) == 0:
                y_pred = self.expression_lambda(*X_in.T)
            else:
                y_pred = self.expression_lambda(*X_in.T, *constants)
            n_samples = X_in.shape[0]
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
            if y_pred.size == 1 and n_samples > 1:
                return np.full((n_samples,), float(y_pred[0]))
            if y_pred.size != n_samples:
                return np.resize(y_pred, n_samples)
            return y_pred

        self._all_constants_values = []
        self.valid_fit = False

        if len(self.constants_symbols) == 0:
            if no_constants_error == "raise":
                raise ValueError("Expression has no constants.")
            diff = pred_function(X) - y[:, 0]
            loss = float(np.mean(diff**2)) if np.all(np.isfinite(diff)) else np.nan
            self._all_constants_values = [(np.array([]), np.array([]), loss)]
            self._assign_fits(self._all_constants_values)
            return self

        p0_noise_kwargs = p0_noise_kwargs or {}
        optimizer_kwargs = optimizer_kwargs or {}
        base_p0 = np.asarray(initial_values, dtype=float) if p0 is None else np.asarray(p0, dtype=float)
        if base_p0.shape[0] != len(self.constants_symbols):
            base_p0 = np.zeros((len(self.constants_symbols),), dtype=float)

        valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y[:, 0])
        X_valid = X[valid_mask]
        y_valid = y[valid_mask, 0]
        if X_valid.shape[0] == 0:
            raise ConvergenceError("No finite samples available for refinement.")

        warnings.filterwarnings("ignore", category=OptimizeWarning)

        # MINPACK's 'lm' errors when there are more constants than data points; 'trf' doesn't.
        curve_fit_method = "trf" if len(self.constants_symbols) > X_valid.shape[0] else "lm"

        for restart_idx in range(max(1, int(n_restarts))):
            p0_run = base_p0.copy()
            if p0_noise is not None:
                if p0_noise == "uniform":
                    low = p0_noise_kwargs.get("low", -1.0)
                    high = p0_noise_kwargs.get("high", 1.0)
                    p0_run = p0_run + np.random.uniform(low=low, high=high, size=p0_run.shape)
                elif p0_noise == "normal":
                    loc = p0_noise_kwargs.get("loc", 0.0)
                    scale = p0_noise_kwargs.get("scale", 1.0)
                    p0_run = p0_run + np.random.normal(loc=loc, scale=scale, size=p0_run.shape)

            try:
                logger.debug("refiner: curve_fit start restart %d", restart_idx)
                t0 = time.monotonic()
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(
                        pred_function, X_valid, y_valid, p0=p0_run, method=curve_fit_method, **optimizer_kwargs
                    )
                    diff = pred_function(X_valid, *popt) - y_valid
                logger.debug("refiner: curve_fit done %.2fs restart %d", time.monotonic() - t0, restart_idx)
                loss = float(np.mean(diff**2)) if np.all(np.isfinite(diff)) else np.nan
                self._all_constants_values.append((np.asarray(popt), np.asarray(pcov), loss))
            except Exception:
                self._all_constants_values.append((np.array([]), np.array([]), np.nan))

        filtered = [
            (constants, cov, loss)
            for constants, cov, loss in self._all_constants_values
            if len(constants) == len(self.constants_symbols)
        ]
        self._assign_fits(filtered)

        if not self.valid_fit and converge_error == "raise":
            raise ConvergenceError("The optimization did not converge.")
        return self

    def predict(self, X: np.ndarray | torch.Tensor, nth_best_constants: int = 0) -> np.ndarray:
        if self.expression_lambda is None:
            raise RuntimeError("Refiner has no compiled expression lambda.")
        constants_values = self._all_constants_values[nth_best_constants][0]
        X = _require_input_set_width(X, self.n_variables)
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        with np.errstate(all="ignore"):
            if len(constants_values) == 0:
                y = self.expression_lambda(*X_np.T)
            else:
                y = self.expression_lambda(*X_np.T, *constants_values)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if y.shape[0] == 1 and X_np.shape[0] > 1:
            y = np.repeat(y, X_np.shape[0], axis=0)
        return y

    def transform(
        self,
        expression: list[str],
        nth_best_constants: int = 0,
        return_prefix: bool = False,
        precision: int = 2,
        variable_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[str] | str:
        constants_values = np.asarray(self._all_constants_values[nth_best_constants][0], dtype=float)
        out_tokens = list(expression)
        if constants_values.size:
            rounded = np.round(constants_values, precision)
            const_idx = 0
            for idx, token in enumerate(out_tokens):
                if (
                    token == "<constant>"
                    or token.startswith("C_")
                    or token in self.constants_symbols
                    or _is_numeric_token(token)
                ):
                    if const_idx < len(rounded):
                        out_tokens[idx] = str(float(rounded[const_idx]))
                        const_idx += 1
        if variable_mapping:
            out_tokens = [variable_mapping.get(token, token) for token in out_tokens]
        if return_prefix:
            return out_tokens
        return self.simplipy_engine.prefix_to_infix(out_tokens, **kwargs)
