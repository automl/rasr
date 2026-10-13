"""Skeleton sampling adapted from https://github.com/psaegert/flash-ansr."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.stats as stats
import simplipy as sp


class NoValidSampleFoundError(Exception):
    pass


class ExpressionSampler:
    """Sample prefix expressions using the configured operator priors."""

    def __init__(
        self,
        simplipy_engine: sp.SimpliPyEngine,
        expression_cfg: dict[str, Any],
        variables: list[str],
        operator_arity: dict[str, int],
        binary_operator_weights: dict[str, float],
        unary_operator_weights: dict[str, float],
        sig_figs: int = 3,
        retries: int = 0,
        operator_meta_enabled: bool = False,
        concentration_dist: str = "skew_normal",
        concentration_loc: float = 30.5,
        concentration_scale: float = 12.0,
        concentration_skew: float = -5.2,
        direct_constants: bool = False,
        direct_constant_probability_std: float = 0.6,
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.expression_cfg = expression_cfg
        self.variables = variables
        self.n_variables = len(variables)
        self.operator_arity = operator_arity
        self.binary_operator_weights = binary_operator_weights
        self.unary_operator_weights = unary_operator_weights
        self.retries = retries
        self._sig_figs = int(sig_figs)
        self._use_operator_meta = bool(operator_meta_enabled)
        self._concentration_dist = concentration_dist
        self._concentration_loc = float(concentration_loc)
        self._concentration_scale = float(concentration_scale)
        self._concentration_skew = float(concentration_skew)
        self._direct_constants = bool(direct_constants)
        self._direct_constant_probability_std = float(direct_constant_probability_std)

        self.unary_operators = [op for op, arity in operator_arity.items() if arity == 1]
        self.binary_operators = [op for op, arity in operator_arity.items() if arity == 2]

        self.unary_operator_probs_base = self._build_probability_vector(self.unary_operators, unary_operator_weights)
        self.binary_operator_probs_base = self._build_probability_vector(self.binary_operators, binary_operator_weights)
        self.unary_operator_probs = self.unary_operator_probs_base.copy()
        self.binary_operator_probs = self.binary_operator_probs_base.copy()

        max_binary_ops = (
            self.expression_cfg["max_binary_ops_base"]
            + self.expression_cfg["max_binary_ops_per_var"] * self.n_variables
        )
        self.binary_distribution = self._generate_ubi_dist(max_binary_ops, 1, 0, 1)

    @classmethod
    def from_config(cls, config: dict) -> "ExpressionSampler":
        expression_cfg = config["prior"]["expressions"]
        variables = config["tokenizer"]["variables"]
        retries = int(config["data_generation"]["sampling_retries"]["expressions"])
        sig_figs = int(expression_cfg["constants"]["mantissa_sig_figs"])
        simplipy_engine = sp.SimpliPyEngine.load(config["data_generation"]["simplipy_engine"], install=True)

        operator_meta_cfg = expression_cfg.get("operator_meta", {})
        operator_meta_enabled = bool(operator_meta_cfg.get("enabled", False))
        concentration_dist = str(operator_meta_cfg.get("concentration_dist", "skew_normal"))
        concentration_loc = float(operator_meta_cfg.get("concentration_loc", 30.5))
        concentration_scale = float(operator_meta_cfg.get("concentration_scale", 12.0))
        concentration_skew = float(operator_meta_cfg.get("concentration_skew", -5.2))

        constants_cfg = expression_cfg.get("constants", {})
        direct_constants = bool(constants_cfg.get("direct", False))
        direct_constant_probability_std = float(constants_cfg.get("direct_probability_std", 0.6))

        return cls(
            simplipy_engine=simplipy_engine,
            expression_cfg=expression_cfg,
            variables=variables,
            operator_arity=expression_cfg["operator_arity"],
            binary_operator_weights=expression_cfg["binary_operator_weights"],
            unary_operator_weights=expression_cfg["unary_operator_weights"],
            retries=retries,
            sig_figs=sig_figs,
            operator_meta_enabled=operator_meta_enabled,
            concentration_dist=concentration_dist,
            concentration_loc=concentration_loc,
            concentration_scale=concentration_scale,
            concentration_skew=concentration_skew,
            direct_constants=direct_constants,
            direct_constant_probability_std=direct_constant_probability_std,
        )
    
    @staticmethod
    def _generate_ubi_dist(
        max_n_operators: int, n_leaves: int, n_unary_operators: int, n_binary_operators: int
    ) -> list[list[int]]:
        """Pre-compute the number of possible trees for operator/leaf counts."""
        D: list[list[int]] = []
        D.append([0] + ([n_leaves ** i for i in range(1, 2 * max_n_operators + 1)]))
        for n in range(1, 2 * max_n_operators + 1):
            s = [0]
            for e in range(1, 2 * max_n_operators - n + 1):
                s.append(
                    n_leaves * s[e - 1]
                    + n_unary_operators * D[n - 1][e]
                    + n_binary_operators * D[n - 1][e + 1]
                )
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
        return D

    @staticmethod
    def _build_probability_vector(operators: Sequence[str], weights: dict[str, float]) -> np.ndarray:
        probs = np.array([weights.get(op, 0) for op in operators], dtype=np.float64)
        return probs / probs.sum()

    def _resample_operator_probs(self) -> None:
        """Resample operator probabilities from Dirichlet(s * base_probs) with a random concentration s.

        Low s yields specialist expressions, high s near-baseline operator usage.
        """
        if not self._use_operator_meta:
            return
        if self._concentration_dist == "skew_normal":
            from scipy.stats import skewnorm
            s = skewnorm.rvs(self._concentration_skew, loc=self._concentration_loc, scale=self._concentration_scale)
        elif self._concentration_dist == "log_normal":
            s = np.exp(np.random.normal(self._concentration_loc, self._concentration_scale))
        elif self._concentration_dist == "log_uniform":
            s = np.exp(np.random.uniform(self._concentration_loc, self._concentration_loc + self._concentration_scale))
        else:
            raise ValueError(f"Unknown concentration_dist: {self._concentration_dist!r}")
        s = max(float(s), 1e-6)
        if len(self.unary_operators) > 0:
            self.unary_operator_probs = np.random.dirichlet(s * self.unary_operator_probs_base)
        self.binary_operator_probs = np.random.dirichlet(s * self.binary_operator_probs_base)

    def _sample_next_pos_binary(self, n_empty_nodes: int, n_ops_remaining: int) -> int:
        if n_empty_nodes >= len(self.binary_distribution):
            self.binary_distribution = self._generate_ubi_dist(n_empty_nodes + 1, 1, 0, 1)
        probs = np.array([
            self.binary_distribution[n_empty_nodes - i + 1][n_ops_remaining - 1]
            for i in range(n_empty_nodes)
        ], dtype=np.float64)
        probs /= self.binary_distribution[n_empty_nodes][n_ops_remaining]
        return int(np.random.choice(n_empty_nodes, p=probs))

    def _get_leaves(self, t_leaves: int, n_unique_symbols: int | None = None) -> list[str]:
        """Return shuffled leaf tokens: unique variables plus padding (variables, or constants in direct mode)."""
        if n_unique_symbols is None:
            n_unique_symbols = np.random.randint(1, min(t_leaves, self.n_variables) + 1)
        else:
            n_unique_symbols = int(n_unique_symbols)
            if n_unique_symbols < 1:
                raise ValueError("n_unique_symbols must be >= 1 when provided.")
            n_unique_symbols = min(n_unique_symbols, t_leaves, self.n_variables)
        unique_symbols = list(self.variables[:n_unique_symbols])
        n_remaining = t_leaves - len(unique_symbols)

        if n_remaining <= 0:
            extra: list[str] = []
        elif self._direct_constants:
            constant_probability = float(
                stats.truncnorm.rvs(
                    0.0,
                    1.0 / self._direct_constant_probability_std,
                    loc=0.0,
                    scale=self._direct_constant_probability_std,
                    size=1,
                )[0]
            )
            is_constant = np.random.random(size=n_remaining) < constant_probability
            n_constants = int(np.sum(is_constant))
            extra_arr = np.empty(n_remaining, dtype=object)
            if n_constants > 0:
                extra_arr[is_constant] = "<constant>"
            n_variable_slots = n_remaining - n_constants
            if n_variable_slots > 0:
                extra_arr[~is_constant] = np.random.choice(unique_symbols, n_variable_slots, replace=True)
            extra = extra_arr.tolist()
        else:
            extra = np.random.choice(unique_symbols, size=n_remaining, replace=True).tolist()

        leaves = unique_symbols + extra
        np.random.shuffle(leaves)
        return leaves

    _NO_WRAP_UNARY: frozenset[str] = frozenset({"neg"})

    def _insert_affine_wrappers(
        self,
        tokens: list[str],
        idx: int,
        arity_dict: dict[str, int],
        *,
        _allow_leaf_scale: bool = True,
        _inside_product: bool = False,
        _inside_sum: bool = False,
    ) -> tuple[list[str], int]:
        """Recursively insert affine ``<constant>`` wrappers into the subtree at *idx*; return (tokens, next_idx).

        The private flags suppress wrappers that would be redundant with one added by the parent.
        """
        token = tokens[idx]
        arity = arity_dict.get(token, 0)
        idx += 1

        if arity == 0:
            if _allow_leaf_scale and token.startswith("x"):
                return ["*", "<constant>", token], idx
            return [token], idx

        if arity == 1:
            wrap = token not in self._NO_WRAP_UNARY
            child, idx = self._insert_affine_wrappers(tokens, idx, arity_dict, _allow_leaf_scale=not wrap)
            if wrap:
                child = ["+", "*", "<constant>"] + child + ["<constant>"]
            return [token] + child, idx

        if token == "/":
            left, idx = self._insert_affine_wrappers(tokens, idx, arity_dict, _allow_leaf_scale=False)
            right, idx = self._insert_affine_wrappers(tokens, idx, arity_dict, _allow_leaf_scale=True)
            left = ["*", "<constant>"] + left
            right = ["+"] + right + ["<constant>"]
        elif token in ("+", "-"):
            left, idx = self._insert_affine_wrappers(
                tokens, idx, arity_dict, _allow_leaf_scale=False, _inside_sum=True
            )
            right, idx = self._insert_affine_wrappers(
                tokens, idx, arity_dict, _allow_leaf_scale=False, _inside_sum=True
            )
            if left[0] not in ("+", "-"):
                left = ["*", "<constant>"] + left
            if right[0] not in ("+", "-"):
                right = ["*", "<constant>"] + right
            result = [token] + left + right
            if not _inside_sum:
                result = ["+"] + result + ["<constant>"]
            return result, idx
        elif token == "*":
            left, idx = self._insert_affine_wrappers(
                tokens, idx, arity_dict, _allow_leaf_scale=False, _inside_product=True
            )
            right, idx = self._insert_affine_wrappers(
                tokens, idx, arity_dict, _allow_leaf_scale=False, _inside_product=True
            )
            result = [token] + left + right
            if not _inside_product:
                result = ["*", "<constant>"] + result
                result = ["+"] + result + ["<constant>"]
            return result, idx
        else:
            left, idx = self._insert_affine_wrappers(tokens, idx, arity_dict)
            right, idx = self._insert_affine_wrappers(tokens, idx, arity_dict)
        result = ["+"] + [token] + left + right + ["<constant>"]
        return result, idx

    @staticmethod
    def _replace_multx_with_binary_product(
        expression: Sequence[str], operator_arity: dict[str, int]
    ) -> list[str]:
        """Rewrite unary multX operators to binary * with <constant>."""
        tokens = list(expression)

        def parse(index: int) -> tuple[list[str], int]:
            token = tokens[index]
            arity = operator_arity.get(token)
            if arity is None or arity == 0:
                return [token], index + 1

            children: list[str] = []
            next_index = index + 1
            for _ in range(arity):
                child_tokens, next_index = parse(next_index)
                children.extend(child_tokens)

            if arity == 1 and token.startswith("mult"):
                return ["*", "<constant>", *children], next_index
            return [token, *children], next_index

        rewritten, final_index = parse(0)
        if final_index != len(tokens):
            raise NoValidSampleFoundError(
                "Invalid prefix expression while replacing multX operators."
            )
        return rewritten

    def _sample_binary_skeleton(self, n_binary_ops: int, n_unique_symbols: int | None = None) -> list[str]:
        """Build a uniformly random binary tree with exactly *n_binary_ops* binary operators."""
        stack: list[str | None] = [None]
        n_empty_nodes = 1
        left_leaves = 0
        total_leaves = 1

        for remaining in range(n_binary_ops, 0, -1):
            position = self._sample_next_pos_binary(n_empty_nodes, remaining)
            operator = str(np.random.choice(self.binary_operators, p=self.binary_operator_probs))
            n_empty_nodes += 1 - position
            total_leaves += 1
            left_leaves += position

            insert_index = [idx for idx, v in enumerate(stack) if v is None][left_leaves]
            stack = stack[:insert_index] + [operator, None, None] + stack[insert_index + 1:]

        leaves = self._get_leaves(t_leaves=total_leaves, n_unique_symbols=n_unique_symbols)
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is None:
                stack = stack[:index] + [leaves.pop()] + stack[index + 1:]
        return stack

    def _insert_unary_ops(self, skeleton: list[str], n_unary_ops: int) -> list[str]:
        """Insert *n_unary_ops* unary operators at random positions, each wrapping the subtree rooted there."""
        if n_unary_ops == 0 or not self.unary_operators:
            return skeleton
        positions = sorted(
            np.random.choice(len(skeleton), size=n_unary_ops, replace=True).tolist(),
            reverse=True,
        )
        result = list(skeleton)
        for pos in positions:
            op = str(np.random.choice(self.unary_operators, p=self.unary_operator_probs))
            result = result[:pos] + [op] + result[pos:]
        return result
    
    def _substitude_constants(
        self, skeleton: Sequence[str], constants: Sequence[float | int]
    ) -> list[str]:
        expression: list[str] = []
        constant_idx = 0
        for token in skeleton:
            if token != "<constant>":
                expression.append(token)
                continue

            if constant_idx >= len(constants):
                raise NoValidSampleFoundError("Not enough constants sampled for expression skeleton.")

            constant_value = constants[constant_idx]
            constant_idx += 1
            if constant_value < 0:
                expression.append("neg")
                expression.append(str(abs(constant_value)))
            else:
                expression.append(str(constant_value))

        if constant_idx != len(constants):
            raise NoValidSampleFoundError("Too many constants sampled for expression skeleton.")

        return expression

    def _ensure_contiguous_variable_indices(self, expression: Sequence[str]) -> list[str]:
        """Reindex used variables to a contiguous prefix of configured variables."""
        used_variables = [var for var in self.variables if var in expression]
        if not used_variables:
            return list(expression)

        variable_mapping = {
            variable: self.variables[new_index]
            for new_index, variable in enumerate(used_variables)
        }
        if all(variable_mapping[variable] == variable for variable in used_variables):
            return list(expression)

        return [variable_mapping.get(token, token) for token in expression]
    
    def sample_skeleton_only(
        self,
        num_unique_symbols: int | None = None,
    ) -> tuple[tuple[str, ...], list[str]]:
        """Sample a skeleton (with ``<constant>`` placeholders) without filling constants."""
        self._resample_operator_probs()
        n_variables = (
            np.random.randint(1, self.n_variables + 1)
            if num_unique_symbols is None
            else min(int(num_unique_symbols), self.n_variables)
        )
        n_binary_ops = int(np.random.randint(
            self.expression_cfg["min_binary_ops_base"]
            + self.expression_cfg["min_binary_ops_per_var"] * n_variables,
            self.expression_cfg["max_binary_ops_base"]
            + self.expression_cfg["max_binary_ops_per_var"] * n_variables
            + 1,
        ))
        n_unary_ops = int(np.random.randint(
            self.expression_cfg["min_unary_ops"],
            self.expression_cfg["max_unary_ops"] + 1,
        ))
        skeleton = self._sample_binary_skeleton(n_binary_ops, n_unique_symbols=n_variables)
        skeleton = self._insert_unary_ops(skeleton, n_unary_ops)

        # Direct-constant mode already placed bare <constant> leaves and must avoid data-informed wrappers.
        arity_dict = self.simplipy_engine.operator_arity
        if not self._direct_constants:
            skeleton, _ = self._insert_affine_wrappers(skeleton, 0, arity_dict)

        try:
            skeleton = self.simplipy_engine.simplify(skeleton, inplace=True, max_pattern_length=4)
        except Exception as e:
            raise NoValidSampleFoundError(f"Failed to simplify skeleton: {skeleton}") from e

        skeleton = self._replace_multx_with_binary_product(skeleton, self.simplipy_engine.operator_arity)

        if any(t in skeleton for t in ['float("inf")', 'float("-inf")', 'float("nan")']):
            raise NoValidSampleFoundError(f"Skeleton contains forbidden tokens: {skeleton}")

        skeleton = self._ensure_contiguous_variable_indices(skeleton)
        variables = [var for var in self.variables if var in skeleton]

        if len(variables) == 0:
            raise NoValidSampleFoundError("Expression contains no variables.")

        if len(skeleton) > self.expression_cfg["max_length"]:
            raise NoValidSampleFoundError(f"Skeleton too long: {skeleton}")

        return tuple(skeleton), variables

    def sample_skeleton_only_with_retries(
        self,
        retries: int | None = None,
        num_unique_symbols: int | None = None,
    ) -> tuple[tuple[str, ...], list[str]]:
        max_retries = self.retries if retries is None else int(retries)
        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                return self.sample_skeleton_only(num_unique_symbols=num_unique_symbols)
            except Exception as error:
                last_error = error
        raise NoValidSampleFoundError("Unable to sample a valid skeleton.") from last_error

    def build_expression_from_skeleton(
        self,
        skeleton: tuple[str, ...] | list[str],
        rounded_constants: list,
        raw_constants: list,
        variables: list[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return ``(expression, executable_expression)`` built with rounded and raw constants, respectively."""
        expression = self._substitude_constants(skeleton, rounded_constants)
        executable_expression = self._substitude_constants(skeleton, raw_constants)
        expression = self._ensure_contiguous_variable_indices(expression)
        executable_expression = self._ensure_contiguous_variable_indices(executable_expression)
        return tuple(expression), tuple(executable_expression)

    @property
    def sig_figs(self) -> int:
        return self._sig_figs

    @property
    def max_length(self) -> int:
        return int(self.expression_cfg["max_length"])
